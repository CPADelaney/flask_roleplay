# context/context_optimization.py

"""
Enhanced context optimization and retrieval system for long-running RPG games.

This module provides advanced context management with:
- Multi-level caching with intelligent eviction
- Temporal relevance decay
- Memory consolidation
- Attention-based retrieval
- Predictive pre-loading
- Token budget management
- Vector-based semantic search integration
"""

import asyncio
import logging
import json
import time
import math
import heapq
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np

# Import from existing modules
from logic.aggregator_sdk import ContextCache, IncrementalContextManager, get_optimized_context
from db.connection import get_db_connection

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Enhanced Incremental Context Manager with Time Decay
# -------------------------------------------------------------------------------

class EnhancedIncrementalContextManager:
    """
    Tracks and manages game context with temporal relevance and prioritization.
    """
    
    def __init__(self):
        self.last_context_hash = None
        self.last_context = None
        self.change_log = []
        
        # Keep track of access frequency for different context items
        self.context_access_patterns = defaultdict(int)
        
        # Track the relevance of context elements over time
        self.context_relevance_scores = {}
        
        # Consolidated memories for long-term storage
        self.consolidated_memories = {}
        
        # Cache for context retrieval operations
        self.context_cache = EnhancedContextCache()
    
    async def get_context(self, user_id, conversation_id, user_input, location=None, include_delta=True):
        """
        Get context with temporal relevance and change tracking.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: Current user input for relevance scoring
            location: Optional current location for context filtering
            include_delta: Whether to include delta changes
            
        Returns:
            Context data with optimization
        """
        cache_key = f"context:{user_id}:{conversation_id}:{location or 'none'}"
        
        async def fetch_context():
            # Get current full context using the existing function
            from logic.aggregator_sdk import get_aggregated_roleplay_context
            current_context = await get_aggregated_roleplay_context(user_id, conversation_id)
            
            # Apply temporal relevance to elements
            self._apply_temporal_relevance(current_context)
            
            # Optimize based on input relevance
            self._score_by_input_relevance(current_context, user_input)
            
            # Apply location-based filtering if location provided
            if location:
                self._filter_by_location(current_context, location)
            
            # Hash for change detection
            current_hash = self._hash_context(current_context)
            
            result = {"full_context": current_context, "is_incremental": False}
            
            # If this is not the first request, check for changes
            if self.last_context_hash:
                # If hash matches, nothing changed
                if current_hash == self.last_context_hash:
                    # Still update access patterns
                    self._update_access_patterns(current_context, user_input)
                    return result
                
                # Something changed, compute a delta
                if include_delta:
                    changes = self._compute_changes(self.last_context, current_context)
                    
                    # Store these changes in change log (limited size)
                    self.change_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "changes": changes
                    })
                    if len(self.change_log) > 20:  # Increased from 10 to 20
                        self.change_log.pop(0)
                    
                    result["delta_context"] = changes
                    result["is_incremental"] = True
                    result["change_log"] = self.change_log
            
            # Update our stored state
            self.last_context = current_context
            self.last_context_hash = current_hash
            
            # Update access patterns
            self._update_access_patterns(current_context, user_input)
            
            return result
        
        # Get from cache with importance based on user input's semantic relevance
        importance = 5.0  # Default importance
        if "quest" in user_input.lower() or "mission" in user_input.lower():
            importance = 8.0
        if "character" in user_input.lower() or "stats" in user_input.lower():
            importance = 7.0
        
        # Attempt to get from cache or fetch fresh
        try:
            return await self.context_cache.get(
                cache_key, 
                fetch_context, 
                cache_level=2,
                importance=importance
            )
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            # Fallback to direct retrieval
            return await fetch_context()
    
    def _apply_temporal_relevance(self, context):
        """
        Apply temporal relevance decay to context elements.
        
        Args:
            context: The context data to modify
        """
        now = datetime.now()
        
        # Apply to events based on recency
        if "events" in context:
            for event in context["events"]:
                # Skip if no timestamp
                if "timestamp" not in event:
                    continue
                
                try:
                    # Parse timestamp
                    event_time = datetime.fromisoformat(event["timestamp"])
                    days_ago = (now - event_time).days
                    
                    # Calculate decay factor (older = less relevant)
                    decay_factor = math.exp(-0.1 * days_ago) if days_ago > 0 else 1.0
                    
                    # Store relevance score
                    event_id = event.get("event_id", hash(json.dumps(event)))
                    self.context_relevance_scores[f"event:{event_id}"] = decay_factor
                    
                    # Mark relevance in the event itself
                    event["relevance_score"] = decay_factor
                except:
                    # Default if timestamp parsing fails
                    event["relevance_score"] = 0.5
        
        # Apply to NPCs based on interaction recency and relationship strength
        if "introduced_npcs" in context:
            for npc in context["introduced_npcs"]:
                npc_id = npc.get("npc_id")
                
                # Calculate base relevance from closeness metric if available
                base_relevance = npc.get("closeness", 50) / 100.0
                
                # Apply additional relevance based on access frequency
                access_bonus = min(0.3, self.context_access_patterns.get(f"npc:{npc_id}", 0) * 0.05)
                
                # Calculate final relevance
                relevance = base_relevance + access_bonus
                
                # Store relevance score
                self.context_relevance_scores[f"npc:{npc_id}"] = relevance
                
                # Mark relevance in the NPC data
                npc["relevance_score"] = relevance
    
    def _score_by_input_relevance(self, context, user_input):
        """
        Score context elements by relevance to current user input.
        
        Args:
            context: The context data to score
            user_input: Current user input for relevance scoring
        """
        # Extract key terms from user input
        input_terms = set(user_input.lower().split())
        
        # Score NPCs by name relevance
        if "introduced_npcs" in context:
            for npc in context["introduced_npcs"]:
                name_terms = set(npc.get("npc_name", "").lower().split())
                
                # Calculate overlap between input terms and name
                overlap = len(input_terms.intersection(name_terms))
                
                # If we found a match, boost relevance
                if overlap > 0:
                    current_score = npc.get("relevance_score", 0.5)
                    npc["relevance_score"] = min(1.0, current_score + (overlap * 0.2))
                    
                    # Also store in our relevance scores
                    npc_id = npc.get("npc_id")
                    if npc_id:
                        self.context_relevance_scores[f"npc:{npc_id}"] = npc["relevance_score"]
        
        # Score quests by relevance to input
        if "quests" in context:
            for quest in context["quests"]:
                name_terms = set(quest.get("quest_name", "").lower().split())
                
                # Calculate overlap
                overlap = len(input_terms.intersection(name_terms))
                
                # If we found a match, mark relevance
                if overlap > 0:
                    relevance = min(1.0, 0.6 + (overlap * 0.2))
                    quest["relevance_score"] = relevance
                    
                    # Store in relevance scores
                    quest_id = quest.get("quest_id")
                    if quest_id:
                        self.context_relevance_scores[f"quest:{quest_id}"] = relevance
                else:
                    quest["relevance_score"] = 0.6  # Default for quests
    
    def _filter_by_location(self, context, location):
        """
        Filter context to prioritize elements relevant to current location.
        
        Args:
            context: The context data to filter
            location: Current location name
        """
        if not location:
            return
        
        # Boost relevance of NPCs in current location
        if "introduced_npcs" in context:
            for npc in context["introduced_npcs"]:
                npc_location = npc.get("current_location", "").lower()
                
                if npc_location and npc_location.lower() == location.lower():
                    # Boost relevance
                    current_score = npc.get("relevance_score", 0.5)
                    npc["relevance_score"] = min(1.0, current_score + 0.3)
    
    def _update_access_patterns(self, context, user_input):
        """
        Update access patterns based on context and user input.
        
        Args:
            context: The context data being accessed
            user_input: Current user input
        """
        # Extract key terms from user input
        input_terms = set(user_input.lower().split())
        
        # Track NPC access patterns
        if "introduced_npcs" in context:
            for npc in context["introduced_npcs"]:
                npc_id = npc.get("npc_id")
                if not npc_id:
                    continue
                
                npc_name = npc.get("npc_name", "").lower()
                
                # Check if NPC name appears in input
                if any(term in npc_name or npc_name in term for term in input_terms):
                    self.context_access_patterns[f"npc:{npc_id}"] += 1
        
        # Track quest access patterns
        if "quests" in context:
            for quest in context["quests"]:
                quest_id = quest.get("quest_id")
                if not quest_id:
                    continue
                
                quest_name = quest.get("quest_name", "").lower()
                
                # Check if quest name appears in input
                if any(term in quest_name or quest_name in term for term in input_terms):
                    self.context_access_patterns[f"quest:{quest_id}"] += 1
        
        # Decay all access counts slightly to prioritize recent access
        for key in list(self.context_access_patterns.keys()):
            self.context_access_patterns[key] *= 0.95
            
            # Remove if count becomes negligible
            if self.context_access_patterns[key] < 0.1:
                del self.context_access_patterns[key]
    
    def _hash_context(self, context):
        """
        Create a hash representation of context to detect changes.
        
        Args:
            context: The context to hash
            
        Returns:
            Hash string
        """
        # Use a more deterministic approach for hashing
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def _compute_changes(self, old_context, new_context):
        """
        Compute what changed between contexts with priority flags.
        
        Args:
            old_context: Previous context state
            new_context: Current context state
            
        Returns:
            Dictionary of changes with priority markers
        """
        changes = {
            "added": {},
            "modified": {},
            "removed": {},
            "high_priority_changes": []
        }
        
        # Check for added or modified items
        for key, value in new_context.items():
            if key not in old_context:
                changes["added"][key] = value
                
                # Some additions are high priority
                if key in ["quests", "current_roleplay", "player_stats"]:
                    changes["high_priority_changes"].append(f"added:{key}")
            elif old_context[key] != value:
                changes["modified"][key] = {
                    "old": old_context[key],
                    "new": value
                }
                
                # Check if this is a high priority change
                if key == "player_stats":
                    changes["high_priority_changes"].append(f"modified:{key}")
                elif key == "quests" and self._has_completed_quest(old_context[key], value):
                    changes["high_priority_changes"].append(f"modified:{key}:quest_completed")
        
        # Check for removed items
        for key in old_context:
            if key not in new_context:
                changes["removed"][key] = old_context[key]
        
        return changes
    
    def _has_completed_quest(self, old_quests, new_quests):
        """Check if any quest has been completed between states"""
        if not isinstance(old_quests, list) or not isinstance(new_quests, list):
            return False
            
        # Create dictionary of old quest statuses
        old_statuses = {q.get("quest_id"): q.get("status") for q in old_quests if "quest_id" in q}
        
        # Check for any quest that changed to "completed"
        for quest in new_quests:
            quest_id = quest.get("quest_id")
            if not quest_id:
                continue
                
            if quest.get("status") == "completed" and old_statuses.get(quest_id) != "completed":
                return True
                
        return False
    
    async def consolidate_memories(self, user_id, conversation_id, days_threshold=3):
        """
        Consolidate old memories to reduce context size.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            days_threshold: Age in days for consolidation
            
        Returns:
            Summary of consolidation
        """
        try:
            # Use journaling system to get older memories
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Calculate the cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Get old journal entries
            cursor.execute("""
                SELECT id, entry_type, entry_text, timestamp
                FROM PlayerJournal
                WHERE user_id=%s AND conversation_id=%s AND timestamp < %s
                ORDER BY timestamp
            """, (user_id, conversation_id, cutoff_date))
            
            entries = cursor.fetchall()
            
            if not entries:
                return {"consolidated": False, "reason": "No old memories to consolidate"}
            
            # Group entries by type
            entry_groups = defaultdict(list)
            for entry_id, entry_type, entry_text, timestamp in entries:
                entry_groups[entry_type].append({
                    "id": entry_id,
                    "text": entry_text,
                    "timestamp": timestamp
                })
            
            # Process each group to create summaries
            summaries = {}
            for entry_type, entries in entry_groups.items():
                # If there's only a few, don't summarize
                if len(entries) < 3:
                    continue
                
                # Create a summary text from all entries
                combined_text = "\n".join([e["text"] for e in entries])
                
                # In a real implementation, you would use an LLM here to generate a summary
                # For now, we'll just create a placeholder summary
                time_range = f"{entries[0]['timestamp']} to {entries[-1]['timestamp']}"
                summary = f"Summary of {len(entries)} {entry_type} entries from {time_range}: {combined_text[:100]}..."
                
                summaries[entry_type] = {
                    "summary": summary,
                    "count": len(entries),
                    "time_range": time_range,
                    "entry_ids": [e["id"] for e in entries]
                }
            
            # Store summarized memories
            for entry_type, summary_data in summaries.items():
                # Add to consolidated memories
                group_key = f"{user_id}:{conversation_id}:{entry_type}"
                self.consolidated_memories[group_key] = summary_data
                
                # Mark original entries as consolidated in database
                # This would involve updating a flag in your database schema
                # For now, let's just log it
                logger.info(f"Consolidated {summary_data['count']} {entry_type} entries for user {user_id}")
            
            return {
                "consolidated": True,
                "summaries": summaries
            }
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return {"consolidated": False, "error": str(e)}
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def get_memory_summaries(self, user_id, conversation_id):
        """
        Get the consolidated memory summaries.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Dictionary of memory summaries
        """
        result = {}
        prefix = f"{user_id}:{conversation_id}:"
        
        for key, summary in self.consolidated_memories.items():
            if key.startswith(prefix):
                entry_type = key[len(prefix):]
                result[entry_type] = summary
        
        return result

# -------------------------------------------------------------------------------
# Enhanced Context Optimizer with Token Budgeting and Vector Integration
# -------------------------------------------------------------------------------

class EnhancedContextOptimizer:
    """
    Advanced context optimization with token budgeting and vector integration.
    """
    
    def __init__(self, vector_db_config=None):
        self.vector_db_config = vector_db_config or {"db_type": "in_memory"}
        self.entity_manager = None
        self.incremental_context_manager = EnhancedIncrementalContextManager()
        self.token_estimator = TokenEstimator()
    
    async def initialize(self, user_id, conversation_id):
        """Initialize the context optimizer"""
        if not self.entity_manager and self.vector_db_config:
            from paste import RPGEntityManager
            self.entity_manager = RPGEntityManager(
                user_id=user_id,
                conversation_id=conversation_id,
                vector_db_config=self.vector_db_config
            )
            await self.entity_manager.initialize()
    
    async def get_optimized_context(
        self, 
        user_id, 
        conversation_id, 
        current_input, 
        location=None, 
        context_budget=4000,
        use_semantic_search=True
    ):
        """
        Retrieve context optimized for token efficiency and relevance.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            current_input: Current user input text
            location: Optional current location
            context_budget: Maximum token budget
            use_semantic_search: Whether to use vector search capabilities
            
        Returns:
            Optimized context dictionary
        """
        await self.initialize(user_id, conversation_id)
        
        # 1. Get basic context with incremental manager
        ctx_result = await self.incremental_context_manager.get_context(
            user_id, conversation_id, current_input, location
        )
        
        base_context = ctx_result["full_context"]
        
        # 2. Get relevant NPCs through our entity manager if available
        relevant_npcs = []
        if self.entity_manager and use_semantic_search:
            try:
                vector_results = await self.entity_manager.get_relevant_entities(
                    query_text=current_input,
                    top_k=5,
                    entity_types=["npc", "location", "memory", "narrative"]
                )
                
                # Extract NPCs from the results
                for item in vector_results:
                    if item.get("metadata", {}).get("entity_type") == "npc":
                        relevant_npcs.append(item["metadata"])
            except Exception as e:
                logger.error(f"Error getting relevant entities: {e}")
        
        # 3. Get recent and relevant memories
        relevant_memories = await self._retrieve_relevant_memories(
            user_id, conversation_id, current_input, limit=5
        )
        
        # 4. Get active conflicts
        active_conflicts = base_context.get("active_conflicts", [])
        
        # 5. Calculate current token usage
        token_usage = {
            "npcs": self.token_estimator.estimate_tokens(base_context.get("introduced_npcs", [])),
            "memories": self.token_estimator.estimate_tokens(relevant_memories),
            "conflicts": self.token_estimator.estimate_tokens(active_conflicts),
            "player_stats": self.token_estimator.estimate_tokens(base_context.get("player_stats", {})),
            "base_context": self.token_estimator.estimate_tokens({
                k: v for k, v in base_context.items() 
                if k not in ["introduced_npcs", "memories", "active_conflicts", "player_stats"]
            })
        }
        
        total_tokens = sum(token_usage.values())
        
        # 6. If over budget, trim less important context
        if total_tokens > context_budget:
            # Prioritize trimming in this order:
            # 1. Reduce memories first
            if token_usage["memories"] > context_budget * 0.2:
                # Cut memories to 20% of budget max
                target_memory_tokens = context_budget * 0.2
                target_memory_ratio = target_memory_tokens / token_usage["memories"]
                
                # Keep most important memories
                memories_with_scores = [(m, m.get("relevance_score", 0.5)) for m in relevant_memories]
                memories_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep memories up to target token count
                kept_memories = []
                kept_tokens = 0
                for memory, score in memories_with_scores:
                    memory_tokens = self.token_estimator.estimate_tokens(memory)
                    if kept_tokens + memory_tokens <= target_memory_tokens:
                        kept_memories.append(memory)
                        kept_tokens += memory_tokens
                
                relevant_memories = kept_memories
                token_usage["memories"] = kept_tokens
            
            # 2. Reduce NPC details next
            if total_tokens - token_usage["memories"] + kept_tokens > context_budget:
                # Sort NPCs by relevance
                npcs = base_context.get("introduced_npcs", [])
                npcs_with_scores = [(npc, npc.get("relevance_score", 0.5)) for npc in npcs]
                npcs_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep most important NPCs but with reduced details for less important ones
                kept_npcs = []
                kept_tokens = 0
                
                target_npc_tokens = context_budget * 0.3  # Allow 30% of budget for NPCs
                
                for npc, score in npcs_with_scores:
                    npc_tokens = self.token_estimator.estimate_tokens(npc)
                    
                    if kept_tokens + npc_tokens <= target_npc_tokens:
                        # Keep full NPC data for most important NPCs
                        kept_npcs.append(npc)
                        kept_tokens += npc_tokens
                    else:
                        # For less important NPCs, keep only essential info
                        simplified_npc = {
                            "npc_id": npc.get("npc_id"),
                            "npc_name": npc.get("npc_name"),
                            "current_location": npc.get("current_location"),
                            "relevance_score": score
                        }
                        
                        simplified_tokens = self.token_estimator.estimate_tokens(simplified_npc)
                        
                        if kept_tokens + simplified_tokens <= target_npc_tokens:
                            kept_npcs.append(simplified_npc)
                            kept_tokens += simplified_tokens
                
                base_context["introduced_npcs"] = kept_npcs
                token_usage["npcs"] = kept_tokens
        
        # 7. Compile final context with priorities
        context = {
            # Core game state always included
            "player_stats": base_context.get("player_stats", {}),
            "current_location": location or base_context.get("current_location", "Unknown"),
            "year": base_context.get("year", "1040"),
            "month": base_context.get("month", "6"),
            "day": base_context.get("day", "15"),
            "time_of_day": base_context.get("time_of_day", "Morning"),
            
            # NPCs sorted by relevance 
            "npcs": self._sort_by_relevance(base_context.get("introduced_npcs", [])),
            
            # Only include most relevant memories
            "memories": relevant_memories,
            
            # Active conflicts are important for story progression
            "active_conflicts": active_conflicts,
            
            # Additional context elements
            "quests": self._sort_by_relevance(base_context.get("quests", [])),
            "social_links": base_context.get("social_links", []),
            
            # Meta information
            "is_delta": ctx_result.get("is_incremental", False),
            "delta_changes": ctx_result.get("delta_context", None),
            "timestamp": datetime.now().isoformat(),
            "token_usage": token_usage,
            "total_tokens": sum(token_usage.values())
        }
        
        return context
    
    def _sort_by_relevance(self, items):
        """Sort items by their relevance score"""
        if not items:
            return []
            
        # Add default relevance if not present
        for item in items:
            if "relevance_score" not in item:
                item["relevance_score"] = 0.5
                
        # Sort by relevance (highest first)
        return sorted(items, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    async def _retrieve_relevant_memories(self, user_id, conversation_id, query_text, limit=5):
        """
        Retrieve relevant memories based on input query.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            query_text: Query text to match against memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get recent journal entries
            cursor.execute("""
                SELECT id, entry_type, entry_text, timestamp
                FROM PlayerJournal
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY timestamp DESC
                LIMIT 20
            """, (user_id, conversation_id))
            
            entries = []
            for entry_id, entry_type, entry_text, timestamp in cursor.fetchall():
                entries.append({
                    "id": entry_id,
                    "type": entry_type,
                    "text": entry_text,
                    "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
                })
            
            # If we have vector capabilities, use semantic search
            if self.entity_manager:
                try:
                    # Get memories through vector search
                    vector_memories = await self.entity_manager.get_relevant_entities(
                        query_text=query_text,
                        top_k=limit,
                        entity_types=["memory"]
                    )
                    
                    # Extract and prioritize these memories
                    if vector_memories:
                        vector_memory_ids = set(m.get("metadata", {}).get("memory_id") for m in vector_memories)
                        
                        # Assign relevance scores based on vector search results
                        for entry in entries:
                            if str(entry["id"]) in vector_memory_ids:
                                # This entry was found by vector search
                                entry["relevance_score"] = 0.8
                            else:
                                # Default relevance for other entries
                                entry["relevance_score"] = 0.4
                except Exception as e:
                    logger.error(f"Error during vector search for memories: {e}")
            
            # If we don't have vector scores, use simple keyword matching
            if not any("relevance_score" in entry for entry in entries):
                query_terms = set(query_text.lower().split())
                
                for entry in entries:
                    # Check term overlap between query and entry text
                    entry_terms = set(entry["text"].lower().split())
                    overlap = len(query_terms.intersection(entry_terms))
                    
                    # Calculate relevance score
                    if overlap > 0:
                        # More overlap = higher score
                        entry["relevance_score"] = min(0.9, 0.5 + (overlap * 0.1))
                    else:
                        # Default score based on recency
                        try:
                            timestamp = datetime.fromisoformat(entry["timestamp"])
                            days_old = (datetime.now() - timestamp).days
                            # Newer = higher score
                            entry["relevance_score"] = max(0.1, 0.5 - (days_old * 0.1))
                        except:
                            entry["relevance_score"] = 0.3
            
            # Sort by relevance and limit
            entries.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return entries[:limit]
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

# -------------------------------------------------------------------------------
# Token Estimator
# -------------------------------------------------------------------------------

class TokenEstimator:
    """
    Utility class to estimate token usage for context objects.
    
    This is a simple estimator that uses general heuristics.
    For production, consider using a proper tokenizer like TikToken.
    """
    
    def __init__(self):
        # Average tokens per character for English text
        self.tokens_per_char = 1/4
        
        # Cache of token estimates
        self.token_cache = {}
    
    def estimate_tokens(self, obj):
        """
        Estimate the number of tokens for a given object.
        
        Args:
            obj: Object to estimate token count for
            
        Returns:
            Estimated token count
        """
        # Check if we have a cached result
        cache_key = self._get_cache_key(obj)
        if cache_key and cache_key in self.token_cache:
            return self.token_cache[cache_key]
        
        if obj is None:
            return 0
        
        # Different handling based on object type
        if isinstance(obj, str):
            tokens = len(obj) * self.tokens_per_char
        elif isinstance(obj, (int, float, bool)):
            tokens = 1
        elif isinstance(obj, list):
            tokens = sum(self.estimate_tokens(item) for item in obj)
        elif isinstance(obj, dict):
            # Add tokens for keys and values
            tokens = sum(
                self.estimate_tokens(k) + self.estimate_tokens(v)
                for k, v in obj.items()
            )
        else:
            # Try to convert to string for other types
            try:
                tokens = len(str(obj)) * self.tokens_per_char
            except:
                tokens = 10  # Default if we can't estimate
        
        # Round up to ensure we don't underestimate
        result = math.ceil(tokens)
        
        # Cache result if we have a cache key
        if cache_key:
            self.token_cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, obj):
        """
        Try to get a cache key for the object.
        
        Args:
            obj: Object to create cache key for
            
        Returns:
            Cache key string or None if not cacheable
        """
        try:
            if isinstance(obj, (str, int, float, bool)):
                return f"{type(obj).__name__}:{str(obj)[:100]}"
            elif isinstance(obj, dict) and "id" in obj:
                # For dictionaries with IDs, use the ID
                return f"dict:{obj['id']}"
            
            # For complex objects, not cacheable
            return None
        except:
            return None
    
    def clear_cache(self):
        """Clear the token estimation cache"""
        self.token_cache.clear()

# -------------------------------------------------------------------------------
# Integration with vector database from paste.txt
# -------------------------------------------------------------------------------

async def get_vector_enhanced_context(
    user_id, 
    conversation_id, 
    input_text, 
    vector_db_config=None
):
    """
    Get enhanced context using vector database integration.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        input_text: Current user input
        vector_db_config: Optional vector database configuration
        
    Returns:
        Enhanced context with semantic search results
    """
    # Import from paste.txt
    from paste import RPGEntityManager
    
    # Default config if none provided
    if vector_db_config is None:
        vector_db_config = {"db_type": "in_memory"}
    
    try:
        # Initialize entity manager
        entity_manager = RPGEntityManager(
            user_id=user_id,
            conversation_id=conversation_id,
            vector_db_config=vector_db_config
        )
        
        await entity_manager.initialize()
        
        # Get context using vector search
        vector_context = await entity_manager.get_context_for_input(input_text)
        
        # Get basic context using our enhanced manager
        context_manager = EnhancedIncrementalContextManager()
        basic_context = await context_manager.get_context(
            user_id, conversation_id, input_text
        )
        
        # Merge contexts with priority
        merged_context = merge_contexts(basic_context["full_context"], vector_context)
        
        return {
            "context": merged_context,
            "vector_search_used": True,
            "is_delta": basic_context.get("is_incremental", False),
            "delta_changes": basic_context.get("delta_context")
        }
    except Exception as e:
        # Fallback to basic context
        logger.error(f"Error getting vector enhanced context: {e}, falling back to basic context")
        
        # Use the existing optimized context function as fallback
        from logic.aggregator_sdk import get_optimized_context
        return await get_optimized_context(user_id, conversation_id, input_text)
    finally:
        # Cleanup
        if 'entity_manager' in locals():
            await entity_manager.close()

def merge_contexts(basic_context, vector_context):
    """
    Merge basic and vector contexts with priority handling.
    
    Args:
        basic_context: Basic context from traditional methods
        vector_context: Context from vector search
        
    Returns:
        Merged context
    """
    # Start with the basic context
    merged = basic_context.copy() if isinstance(basic_context, dict) else {}
    
    # Enhance with vector search results where appropriate
    if "npcs" in vector_context:
        # Add any NPCs from vector search not already in basic context
        basic_npc_ids = {npc.get("npc_id") for npc in merged.get("introduced_npcs", [])}
        vector_npcs = vector_context.get("npcs", [])
        
        for npc in vector_npcs:
            npc_id = npc.get("npc_id")
            if npc_id and npc_id not in basic_npc_ids:
                if "introduced_npcs" not in merged:
                    merged["introduced_npcs"] = []
                merged["introduced_npcs"].append(npc)
                basic_npc_ids.add(npc_id)
    
    # Add memories from vector search
    if "memories" in vector_context:
        vector_memories = vector_context.get("memories", [])
        
        # Create a set of existing memory IDs to avoid duplicates
        memory_ids = {m.get("memory_id") for m in merged.get("memories", [])}
        
        # Add unique memories
        for memory in vector_memories:
            memory_id = memory.get("memory_id")
            if memory_id and memory_id not in memory_ids:
                if "memories" not in merged:
                    merged["memories"] = []
                merged["memories"].append(memory)
                memory_ids.add(memory_id)
    
    # Add locations from vector search
    if "locations" in vector_context:
        vector_locations = vector_context.get("locations", [])
        
        # Create a set of existing location IDs to avoid duplicates
        location_ids = {loc.get("location_id") for loc in merged.get("locations", [])}
        
        # Add unique locations
        for location in vector_locations:
            location_id = location.get("location_id")
            if location_id and location_id not in location_ids:
                if "locations" not in merged:
                    merged["locations"] = []
                merged["locations"].append(location)
                location_ids.add(location_id)
    
    # Add narrative elements from vector search
    if "narratives" in vector_context:
        if "narratives" not in merged:
            merged["narratives"] = []
        
        # Just append all narrative elements, duplicates are less problematic here
        merged["narratives"].extend(vector_context.get("narratives", []))
    
    return merged

# -------------------------------------------------------------------------------
# Predictive Pre-loading
# -------------------------------------------------------------------------------

class PredictiveContextLoader:
    """
    Predicts and pre-loads context that might be needed based on game state.
    """
    
    def __init__(self):
        self.context_optimizer = EnhancedContextOptimizer()
        self.preloaded_contexts = {}
        self.location_prediction_scores = defaultdict(float)
    
    async def update_location_predictions(self, user_id, conversation_id, current_location):
        """
        Update location prediction scores based on current location.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            current_location: Current location name
        """
        # Get connected locations
        connected_locations = await self._get_connected_locations(
            user_id, conversation_id, current_location
        )
        
        # Decay all existing predictions slightly
        for loc in list(self.location_prediction_scores.keys()):
            self.location_prediction_scores[loc] *= 0.8
            
            # Remove if score becomes too low
            if self.location_prediction_scores[loc] < 0.1:
                del self.location_prediction_scores[loc]
        
        # Update predictions for connected locations
        for loc, connection_strength in connected_locations:
            # Higher connection strength = higher probability of going there
            self.location_prediction_scores[loc] += connection_strength
    
    async def preload_likely_contexts(self, user_id, conversation_id, current_location=None):
        """
        Preload context for likely next locations.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            current_location: Current location name
            
        Returns:
            Dictionary with preloading results
        """
        # Update location predictions
        if current_location:
            await self.update_location_predictions(user_id, conversation_id, current_location)
        
        # Get top predicted locations
        likely_locations = sorted(
            self.location_prediction_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 most likely
        
        results = {}
        
        # For each likely location, preload context
        for location, score in likely_locations:
            if score < 0.3:
                continue  # Skip if prediction score is too low
            
            # Check if we already have a recent preload
            cache_key = f"{user_id}:{conversation_id}:{location}"
            cache_info = self.preloaded_contexts.get(cache_key)
            
            # Preload if we don't have it or it's older than 5 minutes
            if not cache_info or time.time() - cache_info.get("timestamp", 0) > 300:
                try:
                    # Create a minimal input text focused on the location
                    input_text = f"Exploring {location}"
                    
                    # Get optimized context for this location
                    context = await self.context_optimizer.get_optimized_context(
                        user_id, conversation_id, input_text, location,
                        context_budget=2000  # Smaller budget for preloads
                    )
                    
                    # Store in cache
                    self.preloaded_contexts[cache_key] = {
                        "context": context,
                        "timestamp": time.time()
                    }
                    
                    results[location] = {"success": True, "new_preload": True}
                except Exception as e:
                    logger.error(f"Error preloading context for {location}: {e}")
                    results[location] = {"success": False, "error": str(e)}
            else:
                # Already preloaded recently
                results[location] = {"success": True, "new_preload": False}
        
        return results
    
    async def get_preloaded_context(self, user_id, conversation_id, location):
        """
        Get preloaded context for a location if available.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            location: Location name
            
        Returns:
            Preloaded context or None if not available
        """
        cache_key = f"{user_id}:{conversation_id}:{location}"
        cache_info = self.preloaded_contexts.get(cache_key)
        
        if cache_info and time.time() - cache_info.get("timestamp", 0) <= 300:
            return cache_info.get("context")
        
        return None
    
    async def _get_connected_locations(self, user_id, conversation_id, current_location):
        """
        Get locations connected to the current one.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            current_location: Current location name
            
        Returns:
            List of (location_name, connection_strength) tuples
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Query for connections in the database
            # This query would need to be adjusted based on your schema
            cursor.execute("""
                SELECT connected_location, connection_strength
                FROM LocationConnections
                WHERE user_id=%s AND conversation_id=%s AND location_name=%s
            """, (user_id, conversation_id, current_location))
            
            connections = cursor.fetchall()
            
            # If no explicit connections, get all locations
            if not connections:
                cursor.execute("""
                    SELECT location_name
                    FROM Locations
                    WHERE user_id=%s AND conversation_id=%s AND location_name!=%s
                """, (user_id, conversation_id, current_location))
                
                # Assign equal probabilities to all locations
                locations = cursor.fetchall()
                base_strength = 0.5 / max(1, len(locations))
                
                return [(loc[0], base_strength) for loc in locations]
            
            return [(loc, strength) for loc, strength in connections]
        except Exception as e:
            logger.error(f"Error getting connected locations: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

# -------------------------------------------------------------------------------
# Comprehensive Context Service
# -------------------------------------------------------------------------------

class ComprehensiveContextService:
    """
    Main service that coordinates all context optimizations.
    """
    
    def __init__(self, vector_db_config=None):
        self.vector_db_config = vector_db_config or {"db_type": "in_memory"}
        self.context_optimizer = EnhancedContextOptimizer(vector_db_config)
        self.predictive_loader = PredictiveContextLoader()
        self.context_cache = EnhancedContextCache()
    
    async def initialize(self, user_id, conversation_id):
        """Initialize the context service"""
        await self.context_optimizer.initialize(user_id, conversation_id)
    
    async def get_context(
        self, 
        user_id, 
        conversation_id, 
        input_text, 
        location=None, 
        context_budget=4000,
        use_vector_search=True,
        use_preloading=True,
        use_caching=True
    ):
        """
        Get comprehensive optimized context for the current interaction.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            input_text: Current user input
            location: Optional current location
            context_budget: Maximum token budget
            use_vector_search: Whether to use vector search
            use_preloading: Whether to use predictive preloading
            use_caching: Whether to use caching
            
        Returns:
            Optimized context dictionary
        """
        await self.initialize(user_id, conversation_id)
        
        cache_key = f"comprehensive_context:{user_id}:{conversation_id}:{location or 'none'}"
        
        # Try to get from cache if caching is enabled
        if use_caching:
            cache_result = self.context_cache.get_from_cache(cache_key, max_age_seconds=60)
            if cache_result:
                return {**cache_result, "source": "cache"}
        
        # Check for preloaded context if preloading is enabled
        preloaded_context = None
        if use_preloading and location:
            preloaded_context = await self.predictive_loader.get_preloaded_context(
                user_id, conversation_id, location
            )
            
            if preloaded_context:
                # We still need to enhance it with query-specific info
                result = await self._enhance_with_query_specifics(
                    preloaded_context, user_id, conversation_id, input_text
                )
                
                # Cache the result
                if use_caching:
                    self.context_cache.add_to_cache(cache_key, result)
                
                return {**result, "source": "preloaded"}
        
        # Get full optimized context
        context = await self.context_optimizer.get_optimized_context(
            user_id, 
            conversation_id, 
            input_text, 
            location, 
            context_budget,
            use_semantic_search=use_vector_search
        )
        
        # If preloading is enabled, trigger background preloading
        if use_preloading:
            # Don't await - let it run in the background
            asyncio.create_task(
                self.predictive_loader.preload_likely_contexts(
                    user_id, conversation_id, location
                )
            )
        
        # Cache the result
        if use_caching:
            self.context_cache.add_to_cache(cache_key, context)
        
        return {**context, "source": "fresh"}
    
    async def _enhance_with_query_specifics(self, preloaded_context, user_id, conversation_id, input_text):
        """
        Enhance preloaded context with query-specific information.
        
        Args:
            preloaded_context: Previously preloaded context
            user_id: User ID
            conversation_id: Conversation ID
            input_text: Current user input
            
        Returns:
            Enhanced context
        """
        try:
            # Extract key terms from input
            input_terms = set(input_text.lower().split())
            
            # Create simple queries based on input terms
            query_terms = [term for term in input_terms 
                          if len(term) > 3 and term not in ('what', 'when', 'where', 'who', 'how', 'why')]
            
            # If we have specific entities to look for
            if query_terms and hasattr(self.context_optimizer, 'entity_manager'):
                # Get relevant entities based on query
                relevant_entities = await self.context_optimizer.entity_manager.get_relevant_entities(
                    query_text=input_text,
                    top_k=3
                )
                
                # Add to preloaded context if not already present
                for entity in relevant_entities:
                    entity_type = entity.get("metadata", {}).get("entity_type")
                    entity_id = entity.get("metadata", {}).get("entity_id")
                    
                    if entity_type == "npc":
                        # Check if this NPC is already in preloaded context
                        preloaded_npcs = preloaded_context.get("npcs", [])
                        npc_ids = {npc.get("npc_id") for npc in preloaded_npcs}
                        
                        if entity_id not in npc_ids:
                            # Add NPC to context
                            npc_data = entity.get("metadata", {})
                            npc_data["relevance_score"] = entity.get("score", 0.5)
                            preloaded_context.setdefault("npcs", []).append(npc_data)
            
            return preloaded_context
        except Exception as e:
            logger.error(f"Error enhancing preloaded context: {e}")
            return preloaded_context
    
    async def consolidate_old_memories(self, user_id, conversation_id, days_threshold=7):
        """
        Consolidate old memories to reduce context size.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            days_threshold: Age in days for consolidation
            
        Returns:
            Consolidation results
        """
        # Use the incremental context manager's consolidation method
        incremental_manager = EnhancedIncrementalContextManager()
        result = await incremental_manager.consolidate_memories(
            user_id, conversation_id, days_threshold
        )
        
        # Invalidate caches after consolidation
        self.context_cache.invalidate(f"context:{user_id}:{conversation_id}:")
        self.context_cache.invalidate(f"comprehensive_context:{user_id}:{conversation_id}:")
        
        return result
    
    async def trim_to_budget(
        self,
        context, 
        token_budget, 
        prioritize_npcs=True
    ):
        """
        Trim context to fit within token budget with intelligent prioritization.
        
        Args:
            context: Context to trim
            token_budget: Maximum token budget
            prioritize_npcs: Whether to prioritize NPCs over other context
            
        Returns:
            Trimmed context
        """
        estimator = TokenEstimator()
        
        # Estimate current token usage
        current_usage = estimator.estimate_tokens(context)
        
        # If we're already within budget, return as is
        if current_usage <= token_budget:
            return context
        
        # Create a working copy
        trimmed = context.copy()
        
        # Calculate how much we need to reduce
        reduction_needed = current_usage - token_budget
        
        # Define priority order for trimming
        if prioritize_npcs:
            # Prioritize keeping NPCs
            trim_order = [
                ("memories", 0.3),       # Trim up to 30% of memories
                ("quests", 0.2),         # Trim up to 20% of quests
                ("narratives", 0.4),     # Trim up to 40% of narratives
                ("active_conflicts", 0.2), # Trim up to 20% of conflicts
                ("npcs", 0.1)            # Trim up to 10% of NPCs as last resort
            ]
        else:
            # More balanced approach
            trim_order = [
                ("narratives", 0.5),     # Trim up to 50% of narratives
                ("memories", 0.3),       # Trim up to 30% of memories
                ("npcs", 0.2),           # Trim up to 20% of NPCs
                ("quests", 0.2),         # Trim up to 20% of quests
                ("active_conflicts", 0.2) # Trim up to 20% of conflicts
            ]
        
        # Sort items in each category by relevance
        for category, _ in trim_order:
            if category in trimmed and isinstance(trimmed[category], list):
                items = trimmed[category]
                
                # Ensure all items have a relevance score
                for item in items:
                    if "relevance_score" not in item:
                        item["relevance_score"] = 0.5
                
                # Sort by relevance (highest first)
                trimmed[category] = sorted(
                    items, 
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True
                )
        
        # Trim each category in priority order
        for category, max_reduction_ratio in trim_order:
            if reduction_needed <= 0:
                break
                
            if category not in trimmed or not isinstance(trimmed[category], list):
                continue
                
            items = trimmed[category]
            
            # Skip if empty
            if not items:
                continue
            
            # Calculate current tokens for this category
            category_tokens = estimator.estimate_tokens(items)
            
            # Calculate maximum tokens to remove from this category
            max_tokens_to_remove = min(
                reduction_needed,
                int(category_tokens * max_reduction_ratio)
            )
            
            if max_tokens_to_remove <= 0:
                continue
            
            # Remove least relevant items first
            items.sort(key=lambda x: x.get("relevance_score", 0))
            
            tokens_removed = 0
            items_to_keep = []
            
            for item in items:
                item_tokens = estimator.estimate_tokens(item)
                
                if tokens_removed + item_tokens <= max_tokens_to_remove:
                    # Remove this item
                    tokens_removed += item_tokens
                else:
                    # Keep this item
                    items_to_keep.append(item)
            
            # Update the category with kept items
            trimmed[category] = sorted(
                items_to_keep,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )
            
            # Update reduction needed
            reduction_needed -= tokens_removed
        
        return trimmed

# -------------------------------------------------------------------------------
# Main API Functions
# -------------------------------------------------------------------------------

# Create singleton instance for the service
_context_service = None

def get_context_service(vector_db_config=None):
    """
    Get or create the comprehensive context service.
    
    Args:
        vector_db_config: Optional vector database configuration
        
    Returns:
        ComprehensiveContextService instance
    """
    global _context_service
    
    if _context_service is None:
        _context_service = ComprehensiveContextService(vector_db_config)
    
    return _context_service

async def get_comprehensive_context(
    user_id, 
    conversation_id, 
    input_text, 
    location=None,
    context_budget=4000,
    use_vector_search=True,
    use_preloading=True,
    use_caching=True
):
    """
    Main API function to get optimized context for RPG interactions.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        input_text: Current user input
        location: Optional current location
        context_budget: Maximum token budget
        use_vector_search: Whether to use vector search
        use_preloading: Whether to use predictive preloading
        use_caching: Whether to use caching
        
    Returns:
        Optimized context dictionary
    """
    service = get_context_service()
    
    context = await service.get_context(
        user_id=user_id,
        conversation_id=conversation_id,
        input_text=input_text,
        location=location,
        context_budget=context_budget,
        use_vector_search=use_vector_search,
        use_preloading=use_preloading,
        use_caching=use_caching
    )
    
    return context

async def consolidate_memories(user_id, conversation_id, days_threshold=7):
    """
    Consolidate old memories to reduce context size.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        days_threshold: Age in days for consolidation
        
    Returns:
        Consolidation results
    """
    service = get_context_service()
    return await service.consolidate_old_memories(user_id, conversation_id, days_threshold)
