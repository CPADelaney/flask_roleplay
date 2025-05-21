# story_agent/progressive_summarization.py

import os
import logging
import asyncio
import json
import time
import asyncpg
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Database connection
from db.connection import get_db_connection_context, initialize_connection_pool, close_connection_pool

# Try to import OpenAI for summarization
try:
    import openai
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

# Configure logging
logger = logging.getLogger(__name__)

class SummaryLevel:
    """Enumeration of summary levels"""
    DETAILED = 0   # Full details
    CONDENSED = 1  # Somewhat condensed
    SUMMARY = 2    # Brief summary
    HEADLINE = 3   # Just a headline/key point


class NarrativeSummarizer:
    """
    Interface for summarization services
    """
    async def summarize(self, text: str, target_level: int, max_tokens: int = 0) -> str:
        """Summarize text to the given level"""
        raise NotImplementedError("Subclasses must implement summarize")


class OpenAISummarizer(NarrativeSummarizer):
    """
    Summarizer using OpenAI API
    """
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano"):
        if not HAVE_OPENAI:
            raise ImportError("openai is required for OpenAISummarizer")
        self.client = openai.AsyncClient(api_key=api_key)
        self.model = model
    
    async def summarize(self, text: str, target_level: int, max_tokens: int = 0) -> str:
        instructions = {
            SummaryLevel.DETAILED: "Create a slightly condensed version that preserves most details.",
            SummaryLevel.CONDENSED: "Create a condensed version that preserves important details and context.",
            SummaryLevel.SUMMARY: "Create a brief summary that captures the main points and key context.",
            SummaryLevel.HEADLINE: "Create a headline or single sentence that captures the essence."
        }
        instruction = instructions.get(target_level, "Summarize this text appropriately.")
        if max_tokens > 0: instruction += f" Keep the summary under {max_tokens} tokens."
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a narrative summarizer. {instruction}"},
                    {"role": "user", "content": text}
                ],
                max_tokens=max_tokens if max_tokens > 0 else None
            )
            return response.choices[0].message.content if response.choices else ""
        except Exception as e:
            logger.error(f"OpenAI summarization error: {e}", exc_info=True)
            return f"[Error summarizing: {type(e).__name__}]" # Fallback


class RuleSummarizer(NarrativeSummarizer):
    """
    Simple rule-based summarizer (fallback when no OpenAI)
    """
    def __init__(self):
        pass
    
    async def summarize(self, text: str, target_level: int, max_tokens: int = 0) -> str:
        """
        Summarize text using simple rules
        
        Args:
            text: Text to summarize
            target_level: Target summary level (0-3)
            max_tokens: Maximum characters for summary (0 for auto)
            
        Returns:
            Summarized text
        """
        # Convert max_tokens to approximate character count (rough estimate)
        max_chars = max_tokens * 4 if max_tokens > 0 else 0
        
        # Split text into sentences
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        
        if target_level == SummaryLevel.DETAILED:
            # Just keep most sentences
            if max_chars > 0 and len(text) > max_chars:
                # Keep sentences until we hit the limit
                result = ""
                for sentence in sentences:
                    if len(result) + len(sentence) + 2 <= max_chars:
                        result += sentence + ". "
                    else:
                        break
                return result.strip()
            return text
            
        elif target_level == SummaryLevel.CONDENSED:
            # Keep about half the sentences
            keep_count = max(1, len(sentences) // 2)
            # Prefer keeping first and last sentences
            if keep_count >= 3 and len(sentences) >= 3:
                important_indices = [0, len(sentences) // 2, len(sentences) - 1]
                other_indices = [i for i in range(len(sentences)) if i not in important_indices]
                
                # Keep first, middle, last, and select others to reach keep_count
                additional_count = keep_count - 3
                if additional_count > 0:
                    selected_indices = important_indices + other_indices[:additional_count]
                    selected_indices.sort()  # Keep original order
                else:
                    selected_indices = important_indices
                
                selected_sentences = [sentences[i] for i in selected_indices]
            else:
                # Simple approach for short texts
                selected_sentences = sentences[:keep_count]
            
            result = ". ".join(selected_sentences)
            if not result.endswith("."):
                result += "."
                
            return result
            
        elif target_level == SummaryLevel.SUMMARY:
            # Keep just first and last sentences
            if len(sentences) >= 2:
                return f"{sentences[0]}. {sentences[-1]}."
            elif sentences:
                return f"{sentences[0]}."
            else:
                return ""
                
        elif target_level == SummaryLevel.HEADLINE:
            # Keep just first sentence, truncated if needed
            if not sentences:
                return ""
                
            first_sentence = sentences[0]
            if max_chars > 0 and len(first_sentence) > max_chars:
                return first_sentence[:max_chars - 3] + "..."
            return first_sentence
            
        # Default case
        return text


class EventInfo:
    """Information about a story event"""
    
    def __init__(
        self,
        event_id: str,
        event_type: str,
        content: str,
        timestamp: datetime,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_id = event_id
        self.event_type = event_type
        self.content = content
        self.timestamp = timestamp
        self.importance = importance  # 0.0 to 1.0
        self.tags = tags or []
        self.metadata = metadata or {}
        self.summaries: Dict[int, str] = {SummaryLevel.DETAILED: content}
        self.last_accessed = timestamp
        self.access_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata,
            "summaries": self.summaries,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventInfo':
        """Create from dictionary"""
        event = cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data["importance"],
            tags=data["tags"],
            metadata=data["metadata"]
        )
        
        # Restore summaries
        event.summaries = {int(k): v for k, v in data.get("summaries", {}).items()}
        
        # Restore access info
        event.last_accessed = datetime.fromisoformat(data["last_accessed"])
        event.access_count = data["access_count"]
        
        return event
    
    def record_access(self) -> None:
        """Record an access to this event"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_summary(self, level: int) -> str:
        """Get summary at the requested level, if available"""
        return self.summaries.get(level, self.summaries.get(SummaryLevel.DETAILED, self.content))
    
    def has_summary(self, level: int) -> bool:
        """Check if a summary exists at the specified level"""
        return level in self.summaries
    
    def set_summary(self, level: int, summary: str) -> None:
        """Set a summary at the specified level"""
        self.summaries[level] = summary


class StoryArc:
    """Represents a story arc containing related events"""
    
    def __init__(
        self,
        arc_id: str,
        title: str,
        description: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        status: str = "active",
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ):
        self.arc_id = arc_id
        self.title = title
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.status = status  # active, completed, abandoned
        self.importance = importance  # 0.0 to 1.0
        self.tags = tags or []
        self.event_ids: List[str] = []
        self.summaries: Dict[int, str] = {SummaryLevel.DETAILED: description}
        self.last_accessed = datetime.now()
        self.access_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "arc_id": self.arc_id,
            "title": self.title,
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
            "importance": self.importance,
            "tags": self.tags,
            "event_ids": self.event_ids,
            "summaries": self.summaries,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryArc':
        """Create from dictionary"""
        end_date = datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None
        
        arc = cls(
            arc_id=data["arc_id"],
            title=data["title"],
            description=data["description"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=end_date,
            status=data["status"],
            importance=data["importance"],
            tags=data["tags"]
        )
        
        # Restore event IDs
        arc.event_ids = data.get("event_ids", [])
        
        # Restore summaries
        arc.summaries = {int(k): v for k, v in data.get("summaries", {}).items()}
        
        # Restore access info
        arc.last_accessed = datetime.fromisoformat(data["last_accessed"])
        arc.access_count = data["access_count"]
        
        return arc
    
    def add_event(self, event_id: str) -> None:
        """Add an event to this arc"""
        if event_id not in self.event_ids:
            self.event_ids.append(event_id)
    
    def remove_event(self, event_id: str) -> bool:
        """Remove an event from this arc"""
        if event_id in self.event_ids:
            self.event_ids.remove(event_id)
            return True
        return False
    
    def record_access(self) -> None:
        """Record an access to this arc"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_summary(self, level: int) -> str:
        """Get summary at the requested level, if available"""
        return self.summaries.get(level, self.summaries.get(SummaryLevel.DETAILED, self.description))
    
    def has_summary(self, level: int) -> bool:
        """Check if a summary exists at the specified level"""
        return level in self.summaries
    
    def set_summary(self, level: int, summary: str) -> None:
        """Set a summary at the specified level"""
        self.summaries[level] = summary
    
    def complete(self, end_date: Optional[datetime] = None) -> None:
        """Mark this arc as completed"""
        self.status = "completed"
        self.end_date = end_date or datetime.now()


class ProgressiveNarrativeSummarizer:
    def __init__(
        self,
        summarizer: Optional[NarrativeSummarizer] = None,
        db_connection_string: Optional[str] = None # This might now be less relevant if DSN is global
                                                   # or if the app passes a pool reference.
                                                   # For now, keep it to indicate DB usage.
    ):
        self.summarizer = summarizer or RuleSummarizer()
        self.db_connection_string = db_connection_string # Indicates if DB operations should be attempted
        
        self.events: Dict[str, EventInfo] = {}
        self.arcs: Dict[str, StoryArc] = {}
        self.event_arc_map: Dict[str, Set[str]] = {}

        self.age_thresholds = {
            SummaryLevel.CONDENSED: timedelta(days=7),
            SummaryLevel.SUMMARY: timedelta(days=30),
            SummaryLevel.HEADLINE: timedelta(days=90)
        }
        self.recency_weight = 0.6
        self.summary_task: Optional[asyncio.Task] = None
        logger.info("ProgressiveNarrativeSummarizer instance created.")
    
    async def initialize(self) -> None:
        """
        Initialize the summarizer.
        Creates necessary DB tables if db_connection_string is set and loads data.
        It RELIES on the global DB_POOL being initialized by the main application.
        """
        logger.info("Initializing ProgressiveNarrativeSummarizer...")
        if self.db_connection_string:
            try:
                logger.info("ProgressiveNarrativeSummarizer: Ensuring database tables and loading data...")
                # Use a single connection for all setup DB operations
                async with get_db_connection_context() as conn: # This connection is 'conn_setup'
                    # --- Create tables ---
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS narrative_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        importance FLOAT NOT NULL DEFAULT 0.5,
                        tags JSONB DEFAULT '[]',
                        metadata JSONB DEFAULT '{}',
                        summaries JSONB DEFAULT '{}',
                        last_accessed TIMESTAMPTZ NOT NULL,
                        access_count INTEGER NOT NULL DEFAULT 0
                    );''')
                    
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS story_arcs (
                        arc_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        start_date TIMESTAMPTZ NOT NULL,
                        end_date TIMESTAMPTZ,
                        status TEXT NOT NULL DEFAULT 'active',
                        importance FLOAT NOT NULL DEFAULT 0.5,
                        tags JSONB DEFAULT '[]',
                        event_ids JSONB DEFAULT '[]',
                        summaries JSONB DEFAULT '{}',
                        last_accessed TIMESTAMPTZ NOT NULL,
                        access_count INTEGER NOT NULL DEFAULT 0
                    );''')
                    
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS event_arc_relationships (
                        event_id TEXT NOT NULL,
                        arc_id TEXT NOT NULL,
                        PRIMARY KEY (event_id, arc_id),
                        FOREIGN KEY (event_id) REFERENCES narrative_events(event_id) ON DELETE CASCADE,
                        FOREIGN KEY (arc_id) REFERENCES story_arcs(arc_id) ON DELETE CASCADE
                    );''')
                    
                    await conn.execute("CREATE INDEX IF NOT EXISTS narrative_events_timestamp_idx ON narrative_events (timestamp);")
                    await conn.execute("CREATE INDEX IF NOT EXISTS narrative_events_type_idx ON narrative_events (event_type);")
                    await conn.execute("CREATE INDEX IF NOT EXISTS story_arcs_status_idx ON story_arcs (status);")
                    logger.info("ProgressiveNarrativeSummarizer: Database tables checked/created.")
    
                    # --- Load data using the SAME connection ---
                    logger.info("ProgressiveNarrativeSummarizer: Loading data from database...")
                    event_rows = await conn.fetch("SELECT * FROM narrative_events")
                    for row_dict in event_rows:
                        try:
                            event = EventInfo.from_dict(dict(row_dict))
                            self.events[event.event_id] = event
                        except Exception as e_event:
                            logger.error(f"Error processing event row: {row_dict}, error: {e_event}", exc_info=True)
                    
                    arc_rows = await conn.fetch("SELECT * FROM story_arcs") # LINE 447 (approx) - uses conn_setup
                    for row_dict in arc_rows:
                        try:
                            arc = StoryArc.from_dict(dict(row_dict))
                            self.arcs[arc.arc_id] = arc
                        except Exception as e_arc:
                            logger.error(f"Error processing arc row: {row_dict}, error: {e_arc}", exc_info=True)
    
                    rel_rows = await conn.fetch("SELECT event_id, arc_id FROM event_arc_relationships")
                    for row_dict in rel_rows:
                        event_id = row_dict["event_id"]
                        arc_id = row_dict["arc_id"]
                        if event_id not in self.event_arc_map:
                            self.event_arc_map[event_id] = set()
                        self.event_arc_map[event_id].add(arc_id)
                    logger.info(f"ProgressiveNarrativeSummarizer: Loaded {len(self.events)} events, {len(self.arcs)} arcs from DB.")
                # Connection conn_setup is released here
            
            except ConnectionError as ce:
                logger.error(f"PNS: DB ConnectionError during init: {ce}", exc_info=True)
                self.db_connection_string = None
                logger.warning("PNS: Disabling DB features due to initialization error.")
            except Exception as e:
                logger.error(f"PNS: Error during DB setup or data load: {e}", exc_info=True)
                self.db_connection_string = None
                logger.warning("PNS: Disabling DB features due to initialization error.")
        else:
            logger.info("ProgressiveNarrativeSummarizer: Initializing in memory-only mode.")
        
        logger.info("ProgressiveNarrativeSummarizer initialized.")
    
    async def _load_data_from_db(self) -> None: # Does NOT take 'conn' as argument
        """Load data from database. Acquires its own connection."""
        if not self.db_connection_string: # Guard clause
            logger.debug("Skipping _load_data_from_db as db_connection_string is not set.")
            return

        logger.info("ProgressiveNarrativeSummarizer: Loading data from database...")
        # This 'try...except' is specific to this data loading operation
        try:
            async with get_db_connection_context() as conn: # Acquires its own connection
                # Load events
                event_rows = await conn.fetch("SELECT * FROM narrative_events")
                for row_dict in event_rows:
                    try:
                        event = EventInfo.from_dict(dict(row_dict))
                        self.events[event.event_id] = event
                    except Exception as e_event:
                        logger.error(f"Error processing event row: {row_dict}, error: {e_event}", exc_info=True)
                
                # Load arcs
                arc_rows = await conn.fetch("SELECT * FROM story_arcs")
                for row_dict in arc_rows:
                    try:
                        arc = StoryArc.from_dict(dict(row_dict))
                        self.arcs[arc.arc_id] = arc
                    except Exception as e_arc:
                        logger.error(f"Error processing arc row: {row_dict}, error: {e_arc}", exc_info=True)

                # Load relationships
                rel_rows = await conn.fetch("SELECT event_id, arc_id FROM event_arc_relationships")
                for row_dict in rel_rows:
                    event_id = row_dict["event_id"]
                    arc_id = row_dict["arc_id"]
                    if event_id not in self.event_arc_map:
                        self.event_arc_map[event_id] = set()
                    self.event_arc_map[event_id].add(arc_id)
            logger.info(f"ProgressiveNarrativeSummarizer: Loaded {len(self.events)} events, {len(self.arcs)} arcs from DB.")
        except Exception as e: # Catch errors specific to data loading
            logger.error(f"Error loading narrative data from DB: {e}", exc_info=True)
            # This error will propagate up to the caller (initialize method) if not caught here,
            # or you can handle it by setting db_connection_string to None.
            # For now, let it propagate to be caught by initialize's except blocks.
            raise
    
    async def close(self) -> None:
        """Clean up resources, like cancelling background tasks."""
        logger.info("Closing ProgressiveNarrativeSummarizer...")
        if self.summary_task and not self.summary_task.done():
            self.summary_task.cancel()
            try:
                await self.summary_task
                logger.info("Summary processor task cancelled.")
            except asyncio.CancelledError:
                logger.info("Summary processor task was already cancelled or finished.")
            except Exception as e: # Catch other potential errors during await
                logger.error(f"Error awaiting summary task cancellation: {e}", exc_info=True)
        self.summary_task = None
            
        # DO NOT call close_connection_pool() here.
        # The main application (main.py @app.after_serving) is responsible for closing the global pool.
        # This class instance doesn't "own" the pool.
        logger.info("ProgressiveNarrativeSummarizer closed.")
    
    async def start_summary_processor(self, interval: int = 3600) -> None:
        """
        Start background task to generate summaries
        
        Args:
            interval: Time between summary generations in seconds (default: 1 hour)
        """
        if self.summary_task is None:
            self.summary_task = asyncio.create_task(self._summary_loop(interval))
    
    async def _summary_loop(self, interval: int) -> None:
        """Background loop for summary generation"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._generate_missing_summaries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in summary generation: {e}")
    
    async def _generate_missing_summaries(self) -> None:
        """Generate missing summaries based on age thresholds"""
        now = datetime.now()
        
        # Process events
        for event_id, event in self.events.items():
            event_age = now - event.timestamp
            
            # Check each level
            for level, threshold in self.age_thresholds.items():
                if event_age >= threshold and not event.has_summary(level):
                    # Generate summary
                    try:
                        summary = await self.summarizer.summarize(
                            event.content, level
                        )
                        event.set_summary(level, summary)
                        
                        # Save to database if using one
                        if self.db_connection_string:
                            await self._save_event_to_db(event)
                    except Exception as e:
                        logger.error(f"Error generating summary for event {event_id}: {e}")
        
        # Process arcs
        for arc_id, arc in self.arcs.items():
            if arc.status == "completed" and arc.end_date:
                arc_age = now - arc.end_date
                
                # Only summarize completed arcs
                for level, threshold in self.age_thresholds.items():
                    if arc_age >= threshold and not arc.has_summary(level):
                        # Generate summary
                        try:
                            # Get full description for summarization
                            content = self._build_arc_content(arc_id)
                            summary = await self.summarizer.summarize(
                                content, level
                            )
                            arc.set_summary(level, summary)
                            
                            # Save to database if using one
                            if self.db_connection_string:
                                await self._save_arc_to_db(arc)
                        except Exception as e:
                            logger.error(f"Error generating summary for arc {arc_id}: {e}")
    
    def _build_arc_content(self, arc_id: str) -> str:
        """Build full content for an arc by combining events"""
        arc = self.arcs.get(arc_id)
        if not arc:
            return ""
        
        # Get detailed content for all events
        contents = []
        for event_id in arc.event_ids:
            event = self.events.get(event_id)
            if event:
                contents.append(f"Event: {event.event_type}\n{event.content}")
        
        # Add arc description
        full_content = f"Story Arc: {arc.title}\n{arc.description}\n\n"
        full_content += "\n\n".join(contents)
        
        return full_content
    
    async def add_event(
        self,
        event_id: str,
        event_type: str,
        content: str,
        timestamp: Optional[datetime] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        arc_ids: Optional[List[str]] = None
    ) -> EventInfo:
        """
        Add a narrative event
        
        Args:
            event_id: Unique ID for the event
            event_type: Type of event (dialog, action, revelation, etc.)
            content: Text content of the event
            timestamp: When this event occurred
            importance: Importance score (0.0 to 1.0)
            tags: List of tags for categorization
            metadata: Additional metadata for this event
            arc_ids: Optional IDs of story arcs this event belongs to
            
        Returns:
            Created EventInfo object
        """
        # Create event
        event = EventInfo(
            event_id=event_id,
            event_type=event_type,
            content=content,
            timestamp=timestamp or datetime.now(),
            importance=importance,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store event
        self.events[event_id] = event
        
        # Handle arc relationships
        if arc_ids:
            for arc_id in arc_ids:
                await self.add_event_to_arc(event_id, arc_id)
        
        # Save to database if using one
        if self.db_connection_string:
            await self._save_event_to_db(event)
        
        return event
    
    async def _save_event_to_db(self, event: EventInfo) -> None:
        if not self.db_connection_string: return
        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    '''
                    INSERT INTO narrative_events (
                        event_id, event_type, content, timestamp, importance,
                        tags, metadata, summaries, last_accessed, access_count
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (event_id) DO UPDATE SET
                        event_type = EXCLUDED.event_type, content = EXCLUDED.content,
                        timestamp = EXCLUDED.timestamp, importance = EXCLUDED.importance,
                        tags = EXCLUDED.tags, metadata = EXCLUDED.metadata, summaries = EXCLUDED.summaries,
                        last_accessed = EXCLUDED.last_accessed, access_count = EXCLUDED.access_count
                    ''',
                    event.event_id, event.event_type, event.content, event.timestamp,
                    event.importance, json.dumps(event.tags or []), json.dumps(event.metadata or {}),
                    json.dumps({str(k): v for k, v in (event.summaries or {}).items()}), # Ensure summaries is a dict
                    event.last_accessed, event.access_count
                )
        except Exception as e:
            logger.error(f"Failed to save event {event.event_id} to DB: {e}", exc_info=True)
    
    async def add_story_arc(
        self,
        arc_id: str,
        title: str,
        description: str,
        start_date: Optional[datetime] = None,
        status: str = "active",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None
    ) -> StoryArc:
        """
        Add a story arc
        
        Args:
            arc_id: Unique ID for the arc
            title: Title of the arc
            description: Description of the arc
            start_date: When this arc started
            status: Status of the arc (active, completed, abandoned)
            importance: Importance score (0.0 to 1.0)
            tags: List of tags for categorization
            event_ids: Optional list of event IDs that belong to this arc
            
        Returns:
            Created StoryArc object
        """
        # Create arc
        arc = StoryArc(
            arc_id=arc_id,
            title=title,
            description=description,
            start_date=start_date or datetime.now(),
            status=status,
            importance=importance,
            tags=tags or []
        )
        
        # Add events if provided
        if event_ids:
            for event_id in event_ids:
                if event_id in self.events:
                    arc.add_event(event_id)
                    
                    # Update event-arc map
                    if event_id not in self.event_arc_map:
                        self.event_arc_map[event_id] = set()
                    self.event_arc_map[event_id].add(arc_id)
        
        # Store arc
        self.arcs[arc_id] = arc
        
        # Save to database if using one
        if self.db_connection_string:
            await self._save_arc_to_db(arc)
            
            # Save relationships
            await self._save_event_arc_relationships(arc)
        
        return arc
    
    async def _save_arc_to_db(self, arc: StoryArc) -> None:
        """Save arc to database"""
        async with get_db_connection_context() as conn:
            await conn.execute('''
            INSERT INTO story_arcs (
                arc_id, title, description, start_date, end_date, status,
                importance, tags, event_ids, summaries, last_accessed, access_count
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (arc_id) DO UPDATE SET
                title = $2,
                description = $3,
                start_date = $4,
                end_date = $5,
                status = $6,
                importance = $7,
                tags = $8,
                event_ids = $9,
                summaries = $10,
                last_accessed = $11,
                access_count = $12
            ''',
            arc.arc_id,
            arc.title,
            arc.description,
            arc.start_date,
            arc.end_date,
            arc.status,
            arc.importance,
            json.dumps(arc.tags),
            json.dumps(arc.event_ids),
            json.dumps({str(k): v for k, v in arc.summaries.items()}),
            arc.last_accessed,
            arc.access_count
            )
    
    async def _save_event_arc_relationships(self, arc: StoryArc) -> None:
        """Save event-arc relationships to database"""
        async with get_db_connection_context() as conn:
            # Delete existing relationships for this arc
            await conn.execute("DELETE FROM event_arc_relationships WHERE arc_id = $1", arc.arc_id)
            
            # Add new relationships
            for event_id in arc.event_ids:
                await conn.execute('''
                INSERT INTO event_arc_relationships (event_id, arc_id)
                VALUES ($1, $2)
                ON CONFLICT (event_id, arc_id) DO NOTHING
                ''', event_id, arc.arc_id)
    
    async def add_event_to_arc(self, event_id: str, arc_id: str) -> bool:
        """
        Add an event to a story arc
        
        Args:
            event_id: ID of the event
            arc_id: ID of the arc
            
        Returns:
            Whether the operation was successful
        """
        # Check if event and arc exist
        event = self.events.get(event_id)
        arc = self.arcs.get(arc_id)
        
        if not event or not arc:
            return False
        
        # Add event to arc
        arc.add_event(event_id)
        
        # Update event-arc map
        if event_id not in self.event_arc_map:
            self.event_arc_map[event_id] = set()
        self.event_arc_map[event_id].add(arc_id)
        
        # Save to database if using one
        if self.db_connection_string:
            await self._save_arc_to_db(arc)
            
            # Save relationship
            async with get_db_connection_context() as conn:
                await conn.execute('''
                INSERT INTO event_arc_relationships (event_id, arc_id)
                VALUES ($1, $2)
                ON CONFLICT (event_id, arc_id) DO NOTHING
                ''', event_id, arc_id)
        
        return True
    
    async def remove_event_from_arc(self, event_id: str, arc_id: str) -> bool:
        """
        Remove an event from a story arc
        
        Args:
            event_id: ID of the event
            arc_id: ID of the arc
            
        Returns:
            Whether the operation was successful
        """
        # Check if arc exists
        arc = self.arcs.get(arc_id)
        
        if not arc:
            return False
        
        # Remove event from arc
        result = arc.remove_event(event_id)
        
        # Update event-arc map
        if event_id in self.event_arc_map:
            self.event_arc_map[event_id].discard(arc_id)
            if not self.event_arc_map[event_id]:
                del self.event_arc_map[event_id]
        
        # Save to database if using one
        if result and self.db_connection_string:
            await self._save_arc_to_db(arc)
            
            # Remove relationship
            async with get_db_connection_context() as conn:
                await conn.execute('''
                DELETE FROM event_arc_relationships
                WHERE event_id = $1 AND arc_id = $2
                ''', event_id, arc_id)
        
        return result
    
    async def complete_arc(self, arc_id: str, end_date: Optional[datetime] = None) -> bool:
        """
        Mark a story arc as completed
        
        Args:
            arc_id: ID of the arc
            end_date: Optional end date (defaults to now)
            
        Returns:
            Whether the operation was successful
        """
        # Check if arc exists
        arc = self.arcs.get(arc_id)
        
        if not arc:
            return False
        
        # Complete arc
        arc.complete(end_date)
        
        # Save to database if using one
        if self.db_connection_string:
            await self._save_arc_to_db(arc)
        
        return True
    
    async def get_event(self, event_id: str, summary_level: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get an event, optionally with a summary
        
        Args:
            event_id: ID of the event
            summary_level: Optional summary level (0-3)
            
        Returns:
            Event data with summary if requested
        """
        event = self.events.get(event_id)
        
        if not event:
            return None
        
        # Record access
        event.record_access()
        
        # Save updated access stats if using database
        if self.db_connection_string:
            await self._update_event_access_stats(event)
        
        # Get content or summary
        content = event.content
        if summary_level is not None and summary_level > SummaryLevel.DETAILED:
            content = await self._ensure_summary_level(event, summary_level)
        
        # Format result
        result = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "content": content,
            "timestamp": event.timestamp.isoformat(),
            "importance": event.importance,
            "tags": event.tags,
            "metadata": event.metadata,
            "arcs": list(self.event_arc_map.get(event_id, set())),
            "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
        }
        
        return result
    
    async def _update_event_access_stats(self, event: EventInfo) -> None:
        """Update event access statistics in database"""
        async with get_db_connection_context() as conn:
            await conn.execute('''
            UPDATE narrative_events
            SET last_accessed = $1, access_count = $2
            WHERE event_id = $3
            ''', event.last_accessed, event.access_count, event.event_id)
    
    async def _ensure_summary_level(self, event: EventInfo, level: int) -> str:
        """
        Ensure a summary exists at the specified level and return it
        
        Args:
            event: The event
            level: Desired summary level (0-3)
            
        Returns:
            The summary text
        """
        # If summary already exists, return it
        if event.has_summary(level):
            return event.get_summary(level)
        
        # Generate summary
        source_level = SummaryLevel.DETAILED
        source_text = event.content
        
        # Find the closest available summary level below the requested level
        for l in range(level - 1, SummaryLevel.DETAILED, -1):
            if event.has_summary(l):
                source_level = l
                source_text = event.get_summary(l)
                break
        
        # Generate summary
        summary = await self.summarizer.summarize(source_text, level)
        event.set_summary(level, summary)
        
        # Save to database if using one
        if self.db_connection_string:
            await self._save_event_to_db(event)
        
        return summary
    
    async def get_arc(self, arc_id: str, summary_level: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get a story arc, optionally with a summary
        
        Args:
            arc_id: ID of the arc
            summary_level: Optional summary level (0-3)
            
        Returns:
            Arc data with summary if requested
        """
        arc = self.arcs.get(arc_id)
        
        if not arc:
            return None
        
        # Record access
        arc.record_access()
        
        # Save updated access stats if using database
        if self.db_connection_string:
            await self._update_arc_access_stats(arc)
        
        # Get description or summary
        description = arc.description
        if summary_level is not None and summary_level > SummaryLevel.DETAILED:
            description = await self._ensure_arc_summary_level(arc, summary_level)
        
        # Format result
        result = {
            "arc_id": arc.arc_id,
            "title": arc.title,
            "description": description,
            "start_date": arc.start_date.isoformat(),
            "end_date": arc.end_date.isoformat() if arc.end_date else None,
            "status": arc.status,
            "importance": arc.importance,
            "tags": arc.tags,
            "event_ids": arc.event_ids,
            "event_count": len(arc.event_ids),
            "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
        }
        
        return result
    
    async def _update_arc_access_stats(self, arc: StoryArc) -> None:
        """Update arc access statistics in database"""
        async with get_db_connection_context() as conn:
            await conn.execute('''
            UPDATE story_arcs
            SET last_accessed = $1, access_count = $2
            WHERE arc_id = $3
            ''', arc.last_accessed, arc.access_count, arc.arc_id)
    
    async def _ensure_arc_summary_level(self, arc: StoryArc, level: int) -> str:
        """
        Ensure a summary exists at the specified level and return it
        
        Args:
            arc: The story arc
            level: Desired summary level (0-3)
            
        Returns:
            The summary text
        """
        # If summary already exists, return it
        if arc.has_summary(level):
            return arc.get_summary(level)
        
        # Generate summary
        source_level = SummaryLevel.DETAILED
        source_text = arc.description
        
        # For detailed level, build full arc content
        if source_level == SummaryLevel.DETAILED:
            source_text = self._build_arc_content(arc.arc_id)
        
        # Find the closest available summary level below the requested level
        for l in range(level - 1, SummaryLevel.DETAILED, -1):
            if arc.has_summary(l):
                source_level = l
                source_text = arc.get_summary(l)
                break
        
        # Generate summary
        summary = await self.summarizer.summarize(source_text, level)
        arc.set_summary(level, summary)
        
        # Save to database if using one
        if self.db_connection_string:
            await self._save_arc_to_db(arc)
        
        return summary
    
    async def get_recent_events(
        self,
        days: int = 7,
        event_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        summary_level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent events
        
        Args:
            days: Number of days to look back
            event_types: Optional list of event types to filter
            tags: Optional list of tags to filter
            limit: Maximum number of events to return
            summary_level: Optional summary level (0-3)
            
        Returns:
            List of recent events
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if self.db_connection_string:
            # Build query conditions
            conditions = ["timestamp >= $1"]
            params = [cutoff_date]
            
            if event_types:
                placeholders = ",".join(f"${i+2}" for i in range(len(event_types)))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend(event_types)
            
            # Handle tags with JSON containment
            if tags:
                for i, tag in enumerate(tags):
                    conditions.append(f"tags @> $" + str(len(params) + 1) + "::jsonb")
                    params.append(json.dumps([tag]))
            
            # Build the query
            query = f'''
            SELECT *
            FROM narrative_events
            WHERE {" AND ".join(conditions)}
            ORDER BY timestamp DESC
            LIMIT ${len(params) + 1}
            '''
            
            params.append(limit)
            
            # Execute query
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, *params)
                
                events = []
                for row in rows:
                    event = EventInfo.from_dict({
                        "event_id": row["event_id"],
                        "event_type": row["event_type"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "importance": row["importance"],
                        "tags": row["tags"],
                        "metadata": row["metadata"],
                        "summaries": row["summaries"],
                        "last_accessed": row["last_accessed"],
                        "access_count": row["access_count"]
                    })
                    
                    # Record access
                    event.record_access()
                    await self._update_event_access_stats(event)
                    
                    # Get content or summary
                    content = event.content
                    if summary_level is not None and summary_level > SummaryLevel.DETAILED:
                        content = await self._ensure_summary_level(event, summary_level)
                    
                    events.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "content": content,
                        "timestamp": event.timestamp.isoformat(),
                        "importance": event.importance,
                        "tags": event.tags,
                        "arcs": list(self.event_arc_map.get(event.event_id, set())),
                        "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
                    })
                
                return events
        else:
            # In-memory filtering
            filtered_events = []
            
            for event_id, event in self.events.items():
                # Apply filters
                if event.timestamp < cutoff_date:
                    continue
                
                if event_types and event.event_type not in event_types:
                    continue
                
                if tags and not any(tag in event.tags for tag in tags):
                    continue
                
                # Record access
                event.record_access()
                
                # Get content or summary
                content = event.content
                if summary_level is not None and summary_level > SummaryLevel.DETAILED:
                    content = await self._ensure_summary_level(event, summary_level)
                
                filtered_events.append({
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "content": content,
                    "timestamp": event.timestamp.isoformat(),
                    "importance": event.importance,
                    "tags": event.tags,
                    "arcs": list(self.event_arc_map.get(event_id, set())),
                    "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
                })
            
            # Sort by timestamp (most recent first) and apply limit
            filtered_events.sort(key=lambda e: e["timestamp"], reverse=True)
            return filtered_events[:limit]
    
    async def get_active_arcs(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        summary_level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active story arcs
        
        Args:
            tags: Optional list of tags to filter
            limit: Maximum number of arcs to return
            summary_level: Optional summary level (0-3)
            
        Returns:
            List of active story arcs
        """
        if self.db_connection_string:
            # Build query conditions
            conditions = ["status = 'active'"]
            params = []
            
            # Handle tags with JSON containment
            if tags:
                for i, tag in enumerate(tags):
                    conditions.append(f"tags @> $" + str(i + 1) + "::jsonb")
                    params.append(json.dumps([tag]))
            
            # Build the query
            query = f'''
            SELECT *
            FROM story_arcs
            WHERE {" AND ".join(conditions)}
            ORDER BY importance DESC, start_date DESC
            LIMIT ${len(params) + 1}
            '''
            
            params.append(limit)
            
            # Execute query
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, *params)
                
                arcs = []
                for row in rows:
                    arc = StoryArc.from_dict({
                        "arc_id": row["arc_id"],
                        "title": row["title"],
                        "description": row["description"],
                        "start_date": row["start_date"],
                        "end_date": row["end_date"],
                        "status": row["status"],
                        "importance": row["importance"],
                        "tags": row["tags"], 
                        "metadata": row["metadata"],
                        "summaries": row["summaries"],
                        "last_accessed": row["last_accessed"],
                        "access_count": row["access_count"]
                    })
                    
                    # Record access
                    arc.record_access()
                    await self._update_arc_access_stats(arc)
                    
                    # Get description or summary
                    description = arc.description
                    if summary_level is not None and summary_level > SummaryLevel.DETAILED:
                        description = await self._ensure_arc_summary_level(arc, summary_level)
                    
                    arcs.append({
                        "arc_id": arc.arc_id,
                        "title": arc.title,
                        "description": description,
                        "start_date": arc.start_date.isoformat(),
                        "importance": arc.importance,
                        "tags": arc.tags,
                        "event_count": len(arc.event_ids),
                        "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
                    })
                
                return arcs
        else:
            # In-memory filtering
            filtered_arcs = []
            
            for arc_id, arc in self.arcs.items():
                # Apply filters
                if arc.status != "active":
                    continue
                
                if tags and not any(tag in arc.tags for tag in tags):
                    continue
                
                # Record access
                arc.record_access()
                
                # Get description or summary
                description = arc.description
                if summary_level is not None and summary_level > SummaryLevel.DETAILED:
                    description = await self._ensure_arc_summary_level(arc, summary_level)
                
                filtered_arcs.append({
                    "arc_id": arc.arc_id,
                    "title": arc.title,
                    "description": description,
                    "start_date": arc.start_date.isoformat(),
                    "importance": arc.importance,
                    "tags": arc.tags,
                    "event_count": len(arc.event_ids),
                    "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
                })
            
            # Sort by importance (highest first) and apply limit
            filtered_arcs.sort(key=lambda a: (a["importance"], a["start_date"]), reverse=True)
            return filtered_arcs[:limit]
    
    async def get_arc_events(
        self,
        arc_id: str,
        limit: int = 10,
        summary_level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get events for a story arc
        
        Args:
            arc_id: ID of the arc
            limit: Maximum number of events to return
            summary_level: Optional summary level (0-3)
            
        Returns:
            List of events for the arc
        """
        arc = self.arcs.get(arc_id)
        
        if not arc:
            return []
        
        # Get events for this arc
        events = []
        
        for event_id in arc.event_ids[:limit]:
            event = self.events.get(event_id)
            
            if event:
                # Record access
                event.record_access()
                
                # Save updated access stats if using database
                if self.db_connection_string:
                    await self._update_event_access_stats(event)
                
                # Get content or summary
                content = event.content
                if summary_level is not None and summary_level > SummaryLevel.DETAILED:
                    content = await self._ensure_summary_level(event, summary_level)
                
                events.append({
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "content": content,
                    "timestamp": event.timestamp.isoformat(),
                    "importance": event.importance,
                    "tags": event.tags,
                    "summary_level": summary_level if summary_level is not None else SummaryLevel.DETAILED
                })
        
        # Sort by timestamp (oldest first)
        events.sort(key=lambda e: e["timestamp"])
        
        return events
    
    async def get_optimal_narrative_context(
        self,
        query: str,
        max_tokens: int = 4000,
        recency_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get optimal narrative context for a query
        
        Args:
            query: Query text
            max_tokens: Maximum tokens for context
            recency_days: Number of days to consider for recent events
            
        Returns:
            Optimized narrative context
        """
        now = datetime.now()
        cutoff_date = now - timedelta(days=recency_days)
        
        # Allocate token budget
        arc_budget = max_tokens * 0.3  # 30% for arcs
        event_budget = max_tokens * 0.7  # 70% for events
        
        # Get active arcs (summarized)
        active_arcs = await self.get_active_arcs(
            limit=5,
            summary_level=SummaryLevel.SUMMARY
        )
        
        # Calculate tokens used by arcs
        arc_tokens = sum(len(arc["description"]) // 4 for arc in active_arcs)
        
        # Adjust event budget based on actual arc usage
        if arc_tokens < arc_budget:
            event_budget += (arc_budget - arc_tokens)
        
        # Find relevant events
        all_events = []
        
        for event_id, event in self.events.items():
            # Skip older events
            if event.timestamp < cutoff_date:
                continue
                
            # Calculate relevance based on recency and access count
            days_old = (now - event.timestamp).days
            recency_score = max(0, 1 - (days_old / recency_days))
            
            access_score = min(1, event.access_count / 10)  # Normalize
            
            # Get most appropriate summary level based on age
            if days_old < 7:
                summary_level = SummaryLevel.DETAILED
            elif days_old < 14:
                summary_level = SummaryLevel.CONDENSED
            else:
                summary_level = SummaryLevel.SUMMARY
            
            # Get content at appropriate level
            content = event.get_summary(summary_level)
            
            # Calculate token estimate
            token_estimate = len(content) // 4
            
            # Check if query terms appear in content
            query_terms = query.lower().split()
            content_lower = content.lower()
            match_score = sum(1 for term in query_terms if term in content_lower) / len(query_terms) if query_terms else 0
            
            # Calculate combined score
            combined_score = (
                self.recency_weight * recency_score + 
                (1 - self.recency_weight) * access_score +
                0.2 * match_score +  # Boost for query match
                0.1 * event.importance  # Small boost for important events
            )
            
            all_events.append({
                "event_id": event.event_id,
                "event_type": event.event_type,
                "content": content,
                "timestamp": event.timestamp.isoformat(),
                "score": combined_score,
                "token_estimate": token_estimate,
                "summary_level": summary_level
            })
        
        # Sort by score and apply token budget
        all_events.sort(key=lambda e: e["score"], reverse=True)
        
        selected_events = []
        tokens_used = 0
        
        for event in all_events:
            if tokens_used + event["token_estimate"] <= event_budget:
                selected_events.append(event)
                tokens_used += event["token_estimate"]
            else:
                # Try with higher summary level
                if event["summary_level"] < SummaryLevel.HEADLINE:
                    # Get higher summary level
                    new_level = event["summary_level"] + 1
                    e = self.events.get(event["event_id"])
                    
                    if e:
                        # Ensure summary exists
                        summary = await self._ensure_summary_level(e, new_level)
                        
                        # Recalculate token estimate
                        token_estimate = len(summary) // 4
                        
                        if tokens_used + token_estimate <= event_budget:
                            event["content"] = summary
                            event["token_estimate"] = token_estimate
                            event["summary_level"] = new_level
                            selected_events.append(event)
                            tokens_used += token_estimate
        
        # Record access for included events
        for event_data in selected_events:
            event = self.events.get(event_data["event_id"])
            if event:
                event.record_access()
                
                # Save updated access stats if using database
                if self.db_connection_string:
                    await self._update_event_access_stats(event)
        
        return {
            "active_arcs": active_arcs,
            "relevant_events": selected_events,
            "token_usage": {
                "arcs": arc_tokens,
                "events": tokens_used,
                "total": arc_tokens + tokens_used,
                "budget": max_tokens
            },
            "query": query
        }


# Example usage
async def example_usage():
    # Create a summarizer (using RuleSummarizer for the example)
    summarizer = RuleSummarizer()
    
    # Create a narrative summarizer
    narrative = ProgressiveNarrativeSummarizer(summarizer=summarizer)
    
    # Add some events
    event1 = await narrative.add_event(
        event_id="event1",
        event_type="dialogue",
        content="Mistress Victoria smiled warmly as she welcomed me into her office. 'I'm glad you could make it today,' she said, gesturing to the chair across from her desk. 'I've been looking forward to our meeting.' Her voice was gentle but carried an undercurrent of authority that made me immediately sit down. 'I've been reviewing your progress, and I must say, I'm quite pleased with how receptive you've been to our... arrangement.'",
        importance=0.7,
        tags=["intro", "victoria"]
    )
    
    event2 = await narrative.add_event(
        event_id="event2",
        event_type="internal",
        content="I couldn't help but notice how Victoria's office seemed different today. The blinds were partially closed, casting striped shadows across the desk. A new photo had appeared on her bookshelf - Victoria standing beside a woman I didn't recognize, both smiling at some private joke. I wondered about their relationship, but felt uncomfortable asking. Everything about the room seemed designed to put me slightly off balance, to remind me that this was her territory, not mine.",
        importance=0.5,
        tags=["observation", "victoria", "office"]
    )
    
    event3 = await narrative.add_event(
        event_id="event3",
        event_type="dialogue",
        content="'What I'd like to discuss today,' Victoria continued, opening a folder on her desk, 'is how we might deepen our work together.' She looked up, her blue eyes suddenly intense. 'I believe you're ready for the next phase.' My mouth felt dry. 'What exactly does that entail?' I asked, trying to keep my voice steady. Victoria's smile widened slightly. 'Trust, primarily. Your willingness to follow my guidance without questioning every step of the process.' She leaned forward. 'Do you think you can manage that?' The question hung in the air between us, weightier than it should have been.",
        importance=0.8,
        tags=["development", "victoria", "control"]
    )
    
    # Create a story arc
    arc1 = await narrative.add_story_arc(
        arc_id="arc1",
        title="Victoria's Influence",
        description="The growing influence of Mistress Victoria over the player's decisions and perceptions.",
        tags=["victoria", "manipulation", "main_plot"],
        event_ids=["event1", "event2", "event3"]
    )
    
    # Generate summaries
    await narrative._ensure_summary_level(event1, SummaryLevel.CONDENSED)
    await narrative._ensure_summary_level(event1, SummaryLevel.SUMMARY)
    await narrative._ensure_summary_level(event1, SummaryLevel.HEADLINE)
    
    # Display different summary levels
    event = await narrative.get_event("event1", SummaryLevel.DETAILED)
    print("\nDETAILED:")
    print(event["content"])
    
    event = await narrative.get_event("event1", SummaryLevel.CONDENSED)
    print("\nCONDENSED:")
    print(event["content"])
    
    event = await narrative.get_event("event1", SummaryLevel.SUMMARY)
    print("\nSUMMARY:")
    print(event["content"])
    
    event = await narrative.get_event("event1", SummaryLevel.HEADLINE)
    print("\nHEADLINE:")
    print(event["content"])
    
    # Get optimal context
    context = await narrative.get_optimal_narrative_context(
        query="Victoria office meeting",
        max_tokens=1000
    )
    
    print("\nOPTIMAL CONTEXT:")
    print(f"Arcs: {len(context['active_arcs'])}")
    print(f"Events: {len(context['relevant_events'])}")
    print(f"Token usage: {context['token_usage']}")

# RPG-Specific Implementation

class RPGNarrativeManager:
    def __init__(
        self,
        user_id: int, # Should probably be str if your user IDs from DB are SERIAL (int) but used as str elsewhere
        conversation_id: int, # Same as above
        db_connection_string: Optional[str] = None, # Or pass DSN explicitly if preferred
        openai_api_key: Optional[str] = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        current_openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY") # Get from env if not passed

        if current_openai_api_key and HAVE_OPENAI:
            summarizer = OpenAISummarizer(current_openai_api_key)
        else:
            if not HAVE_OPENAI: logger.warning("OpenAI library not found. Using rule-based summarizer.")
            if not current_openai_api_key: logger.warning("OpenAI API key not provided. Using rule-based summarizer.")
            summarizer = RuleSummarizer()
        
        # Use the DSN from db.connection if db_connection_string is not provided explicitly
        # but we still want to use DB if the global one is configured.
        # If db_connection_string is explicitly None, it means "don't use DB for this instance".
        effective_db_string = db_connection_string
        if db_connection_string is None: # If caller didn't specify, check if global DSN exists
            try:
                from db.connection import get_db_dsn # Assuming this function exists
                if get_db_dsn(): # Check if a DSN is configured globally
                     effective_db_string = "USE_GLOBAL_POOL" # Signal to use global pool setup
            except ImportError:
                logger.warning("db.connection.get_db_dsn not found, cannot infer global DSN for RPGNarrativeManager.")
            except EnvironmentError: # If get_db_dsn raises error because no DSN is set
                logger.info("No global DB DSN configured. RPGNarrativeManager will operate in memory-only unless db_connection_string is explicitly passed.")


        self.narrative = ProgressiveNarrativeSummarizer(
            summarizer=summarizer,
            db_connection_string=effective_db_string # Pass the determined DSN or signal
        )
        self.active_arcs: Dict[str, Any] = {} # Store arc data, not just IDs
        logger.info(f"RPGNarrativeManager for user {user_id}, conv {conversation_id} created.")
    
    async def initialize(self) -> None:
        logger.info(f"Initializing RPGNarrativeManager for user {self.user_id}, conv {self.conversation_id}...")
        await self.narrative.initialize() # This will handle its DB setup
        await self.narrative.start_summary_processor()
        
        # Load active arcs
        active_arcs = await self.narrative.get_active_arcs()
        self.active_arcs = {arc["arc_id"]: arc for arc in active_arcs}
    
    async def close(self) -> None:
        logger.info(f"Closing RPGNarrativeManager for user {self.user_id}, conv {self.conversation_id}...")
        await self.narrative.close() # This will cancel its summary_task if started
        logger.info(f"RPGNarrativeManager for user {self.user_id}, conv {self.conversation_id} closed.")
    
    async def add_interaction(
        self,
        content: str,
        npc_name: Optional[str] = None,
        location: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a player-NPC interaction to the narrative history
        
        Args:
            content: Text content of the interaction
            npc_name: Optional name of the NPC involved
            location: Optional location where the interaction occurs
            importance: Importance score (0.0 to 1.0)
            tags: Optional list of additional tags
            
        Returns:
            Event ID
        """
        # Generate event ID
        timestamp = datetime.now()
        event_id = f"interaction_{self.user_id}_{self.conversation_id}_{int(timestamp.timestamp())}"
        
        # Build tags
        event_tags = ["interaction"]
        if npc_name:
            event_tags.append(npc_name.lower().replace(" ", "_"))
        if location:
            event_tags.append(location.lower().replace(" ", "_"))
        if tags:
            event_tags.extend(tags)
        
        # Build metadata
        metadata = {
            "npc_name": npc_name,
            "location": location
        }
        
        # Add event
        await self.narrative.add_event(
            event_id=event_id,
            event_type="interaction",
            content=content,
            timestamp=timestamp,
            importance=importance,
            tags=event_tags,
            metadata=metadata
        )
        
        # Add to appropriate arcs
        await self._add_to_relevant_arcs(event_id, npc_name, tags)
        
        return event_id
    
    async def add_revelation(
        self,
        content: str,
        revelation_type: str,
        importance: float = 0.8,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a character revelation to the narrative history
        
        Args:
            content: Text content of the revelation
            revelation_type: Type of revelation (e.g., dependency, control)
            importance: Importance score (0.0 to 1.0)
            tags: Optional list of additional tags
            
        Returns:
            Event ID
        """
        # Generate event ID
        timestamp = datetime.now()
        event_id = f"revelation_{self.user_id}_{self.conversation_id}_{int(timestamp.timestamp())}"
        
        # Build tags
        event_tags = ["revelation", revelation_type]
        if tags:
            event_tags.extend(tags)
        
        # Build metadata
        metadata = {
            "revelation_type": revelation_type
        }
        
        # Add event
        await self.narrative.add_event(
            event_id=event_id,
            event_type="revelation",
            content=content,
            timestamp=timestamp,
            importance=importance,
            tags=event_tags,
            metadata=metadata
        )
        
        # Add to appropriate arcs
        await self._add_to_relevant_arcs(event_id, None, event_tags)
        
        return event_id
    
    async def add_dream_sequence(
        self,
        content: str,
        symbols: List[str],
        importance: float = 0.7,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a dream sequence to the narrative history
        
        Args:
            content: Text content of the dream
            symbols: List of symbolic elements in the dream
            importance: Importance score (0.0 to 1.0)
            tags: Optional list of additional tags
            
        Returns:
            Event ID
        """
        # Generate event ID
        timestamp = datetime.now()
        event_id = f"dream_{self.user_id}_{self.conversation_id}_{int(timestamp.timestamp())}"
        
        # Build tags
        event_tags = ["dream"]
        event_tags.extend(symbols)
        if tags:
            event_tags.extend(tags)
        
        # Build metadata
        metadata = {
            "symbols": symbols
        }
        
        # Add event
        await self.narrative.add_event(
            event_id=event_id,
            event_type="dream",
            content=content,
            timestamp=timestamp,
            importance=importance,
            tags=event_tags,
            metadata=metadata
        )
        
        # Add to appropriate arcs
        await self._add_to_relevant_arcs(event_id, None, event_tags)
        
        return event_id
    
    async def start_story_arc(
        self,
        title: str,
        description: str,
        arc_type: str,
        npc_names: Optional[List[str]] = None,
        importance: float = 0.6,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a new story arc
        
        Args:
            title: Title of the arc
            description: Description of the arc
            arc_type: Type of arc (main_plot, side_quest, character_development)
            npc_names: Optional list of NPCs involved
            importance: Importance score (0.0 to 1.0)
            tags: Optional list of additional tags
            
        Returns:
            Arc ID
        """
        # Generate arc ID
        timestamp = datetime.now()
        arc_id = f"arc_{self.user_id}_{self.conversation_id}_{int(timestamp.timestamp())}"
        
        # Build tags
        arc_tags = [arc_type]
        if npc_names:
            for npc in npc_names:
                arc_tags.append(npc.lower().replace(" ", "_"))
        if tags:
            arc_tags.extend(tags)
        
        # Add arc
        arc = await self.narrative.add_story_arc(
            arc_id=arc_id,
            title=title,
            description=description,
            start_date=timestamp,
            status="active",
            importance=importance,
            tags=arc_tags
        )
        
        # Add to active arcs
        self.active_arcs[arc_id] = {
            "arc_id": arc_id,
            "title": title,
            "description": description,
            "tags": arc_tags
        }
        
        return arc_id
    
    async def complete_story_arc(
        self,
        arc_id: str,
        conclusion: str
    ) -> bool:
        """
        Complete a story arc
        
        Args:
            arc_id: ID of the arc to complete
            conclusion: Text describing the conclusion
            
        Returns:
            Whether the operation was successful
        """
        # Check if arc exists
        arc = await self.narrative.get_arc(arc_id)
        if not arc:
            return False
        
        # Add conclusion event
        timestamp = datetime.now()
        event_id = f"conclusion_{self.user_id}_{self.conversation_id}_{int(timestamp.timestamp())}"
        
        await self.narrative.add_event(
            event_id=event_id,
            event_type="conclusion",
            content=conclusion,
            timestamp=timestamp,
            importance=0.9,
            tags=["conclusion"] + arc["tags"],
            metadata={"arc_id": arc_id, "arc_title": arc["title"]}
        )
        
        # Add conclusion to arc
        await self.narrative.add_event_to_arc(event_id, arc_id)
        
        # Complete arc
        result = await self.narrative.complete_arc(arc_id)
        
        # Remove from active arcs
        if result and arc_id in self.active_arcs:
            del self.active_arcs[arc_id]
        
        return result
    
    async def _add_to_relevant_arcs(
        self,
        event_id: str,
        npc_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Add an event to relevant story arcs
        
        Args:
            event_id: ID of the event
            npc_name: Optional name of an NPC involved
            tags: Optional list of tags
        """
        # Default empty lists
        if tags is None:
            tags = []
        
        # Find matching arcs
        for arc_id, arc in self.active_arcs.items():
            # Check if this arc is relevant
            relevant = False
            
            # Check NPC match
            if npc_name and npc_name.lower().replace(" ", "_") in arc["tags"]:
                relevant = True
            
            # Check tag overlap
            if not relevant and tags:
                for tag in tags:
                    if tag in arc["tags"]:
                        relevant = True
                        break
            
            # Add to arc if relevant
            if relevant:
                await self.narrative.add_event_to_arc(event_id, arc_id)
    
    async def get_current_narrative_context(
        self,
        input_text: str,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Get optimized narrative context for the current input
        
        Args:
            input_text: Current input text
            max_tokens: Maximum tokens for context
            
        Returns:
            Optimized narrative context
        """
        # Get optimal context based on the input
        context = await self.narrative.get_optimal_narrative_context(
            query=input_text,
            max_tokens=max_tokens
        )
        
        # Format for easy use by agents
        formatted_context = {
            "active_story_arcs": [
                {
                    "title": arc["title"],
                    "description": arc["description"],
                    "status": "active"
                }
                for arc in context["active_arcs"]
            ],
            "relevant_events": [
                {
                    "type": event["event_type"],
                    "content": event["content"],
                    "timestamp": event["timestamp"]
                }
                for event in context["relevant_events"]
            ],
            "token_usage": context["token_usage"]
        }
        
        return formatted_context
    
    async def get_npc_interaction_history(
        self,
        npc_name: str,
        max_events: int = 5,
        summary_level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history of interactions with a specific NPC
        
        Args:
            npc_name: Name of the NPC
            max_events: Maximum number of events to return
            summary_level: Optional summary level (0-3)
            
        Returns:
            List of interactions with the NPC
        """
        # Normalize NPC name for tags
        npc_tag = npc_name.lower().replace(" ", "_")
        
        # Get events with this NPC
        events = await self.narrative.get_recent_events(
            days=90,  # Look back up to 90 days
            tags=[npc_tag],
            limit=max_events,
            summary_level=summary_level
        )
        
        return [
            {
                "type": event["event_type"],
                "content": event["content"],
                "timestamp": event["timestamp"]
            }
            for event in events
        ]
    
    async def get_narrative_stage_summary(
        self,
        stage_name: str,
        summary_level: int = SummaryLevel.SUMMARY
    ) -> Dict[str, Any]:
        """
        Get a summary of a narrative stage
        
        Args:
            stage_name: Name of the narrative stage
            summary_level: Summary level (0-3)
            
        Returns:
            Summary of the narrative stage
        """
        # Find arcs related to this stage
        stage_tag = stage_name.lower().replace(" ", "_")
        
        arcs = await self.narrative.get_active_arcs(
            tags=[stage_tag],
            limit=3,
            summary_level=summary_level
        )
        
        if not arcs:
            # Try to find any completed arcs with this tag
            if self.narrative.db_connection_string:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch('''
                    SELECT arc_id, title, description, status
                    FROM story_arcs
                    WHERE tags @> $1::jsonb AND status = 'completed'
                    ORDER BY end_date DESC
                    LIMIT 3
                    ''', json.dumps([stage_tag]))
                    
                    arcs = []
                    for row in rows:
                        arc = await self.narrative.get_arc(
                            row["arc_id"],
                            summary_level=summary_level
                        )
                        if arc:
                            arcs.append(arc)
        
        # Get relevant events
        events = await self.narrative.get_recent_events(
            days=60,  # Look back 60 days
            tags=[stage_tag],
            limit=5,
            summary_level=summary_level
        )
        
        # Build summary
        summary = {
            "stage_name": stage_name,
            "arcs": [
                {
                    "title": arc["title"],
                    "description": arc["description"],
                    "status": arc["status"]
                }
                for arc in arcs
            ],
            "key_events": [
                {
                    "type": event["event_type"],
                    "content": event["content"]
                }
                for event in events
            ]
        }
        
        return summary
