# context/context_service.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from context.context_config import get_config
from context.unified_cache import context_cache
from context.vector_service import get_vector_service
from context.memory_manager import get_memory_manager
from context.context_manager import get_context_manager
from context.performance import PerformanceMonitor, track_performance

logger = logging.getLogger(__name__)

class ContextService:
    """
    Unified context service that integrates all context components.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.initialized = False
        self.performance_monitor = None
        self.context_manager = None
        self.memory_manager = None
        self.vector_service = None
    
    async def initialize(self):
        """Initialize the context service"""
        if self.initialized:
            return
        
        # Get core components (lazy initialization)
        self.context_manager = get_context_manager()
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        self.initialized = True
        logger.info(f"Initialized context service for user {self.user_id}, conversation {self.conversation_id}")
    
    async def close(self):
        """Close the context service"""
        self.initialized = False
        logger.info(f"Closed context service for user {self.user_id}, conversation {self.conversation_id}")
    
    @track_performance("get_context")
    async def get_context(
        self,
        input_text: str = "",
        location: Optional[str] = None,
        context_budget: int = 4000,
        use_vector_search: Optional[bool] = None,
        use_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Get optimized context for the current interaction
        
        Args:
            input_text: Current user input
            location: Optional current location
            context_budget: Maximum token budget
            use_vector_search: Whether to use vector search
            use_delta: Whether to include delta changes
            
        Returns:
            Optimized context
        """
        # Initialize if needed
        await self.initialize()
        
        # Check if vector search should be used
        if use_vector_search is None:
            use_vector_search = self.config.is_enabled("use_vector_search")
        
        # Cache key based on parameters
        cache_key_parts = [
            f"context:{self.user_id}:{self.conversation_id}",
            f"loc:{location or 'none'}",
            f"vec:{use_vector_search}"
        ]
        
        # Add input text hash if not empty
        if input_text:
            import hashlib
            text_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]
            cache_key_parts.append(f"input:{text_hash}")
        
        cache_key = ":".join(cache_key_parts)
        
        # Function to fetch context if not in cache
        async def fetch_context():
            # Record start time
            start_time = time.time()
            
            # Get base context
            context = await self._get_base_context(location)
            
            # Add memories if available
            if self.config.get("features", "use_memories", True):
                memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
                
                if input_text:
                    # Search memories relevant to input
                    memories = await memory_manager.search_memories(
                        query_text=input_text,
                        limit=5,
                        use_vector=use_vector_search
                    )
                else:
                    # Get recent memories
                    memories = await memory_manager.get_recent_memories(
                        days=3, 
                        limit=5
                    )
                
                # Convert to dictionaries
                memory_dicts = []
                for memory in memories:
                    memory_dicts.append({
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "created_at": memory.created_at.isoformat(),
                        "importance": memory.importance,
                        "tags": memory.tags
                    })
                
                context["memories"] = memory_dicts
            
            # Add vector-based content if enabled
            if use_vector_search:
                vector_service = await get_vector_service(self.user_id, self.conversation_id)
                vector_context = await vector_service.get_context_for_input(
                    input_text=input_text,
                    current_location=location
                )
                
                # Merge vector results
                if "npcs" in vector_context and vector_context["npcs"]:
                    context["npcs"] = vector_context["npcs"]
                if "locations" in vector_context and vector_context["locations"]:
                    context["locations"] = vector_context["locations"]
                if "narratives" in vector_context and vector_context["narratives"]:
                    context["narratives"] = vector_context["narratives"]
            
            # Trim to budget if needed
            token_usage = self._estimate_token_usage(context)
            total_tokens = sum(token_usage.values())
            
            if total_tokens > context_budget:
                context
