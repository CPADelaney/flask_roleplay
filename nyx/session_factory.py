# nyx/session_factory.py

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
import weakref

from nyx.agent_session import NyxAgentSession
from nyx.core.brain.base import NyxBrain

logger = logging.getLogger(__name__)

class NyxSessionFactory:
    """
    Factory class for creating and managing Nyx agent sessions.
    
    This class:
    1. Creates and tracks agent sessions
    2. Manages resource allocation
    3. Performs maintenance and cleanup
    4. Provides session statistics
    5. Handles cross-session intelligence
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the session factory."""
        # Active sessions by user_id and conversation_id
        self.sessions = {}
        
        # Session capacity and limits
        self.max_sessions_per_user = 10
        self.max_total_sessions = 1000
        self.session_timeout = 3600  # 1 hour
        
        # Resource monitoring
        self.resource_usage = {
            "memory": {},
            "cpu": {},
            "sessions": {}
        }
        
        # Maintenance task
        self.maintenance_task = None
        self.running = False
        
        # Stats
        self.stats = {
            "total_sessions_created": 0,
            "total_sessions_archived": 0,
            "peak_concurrent_sessions": 0
        }
        
        # Central brain reference
        self.central_brain_instances = {}
    
    async def get_session(
        self, 
        user_id: int, 
        conversation_id: int,
        create_if_missing: bool = True,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Optional[NyxAgentSession]:
        """
        Get an existing session or create a new one.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            create_if_missing: Whether to create a session if not found
            initial_context: Optional initial context for new sessions
            
        Returns:
            Agent session instance or None if not found and create_if_missing=False
        """
        session_key = f"{user_id}_{conversation_id}"
        
        # Check if session exists
        if session_key in self.sessions:
            return self.sessions[session_key]
        
        # Don't create if not requested
        if not create_if_missing:
            return None
        
        # Check limits before creating
        if not self._check_session_limits(user_id):
            logger.warning(f"Session limits reached for user {user_id}")
            return None
        
        # Create new session
        try:
            session = NyxAgentSession(user_id, conversation_id)
            await session.initialize(initial_context)
            
            # Store session
            self.sessions[session_key] = session
            
            # Update stats
            self.stats["total_sessions_created"] += 1
            self._update_concurrent_stats()
            
            # Start maintenance task if not running
            self._ensure_maintenance_running()
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    async def close_session(self, user_id: int, conversation_id: int) -> bool:
        """
        Close and archive a session.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Success status
        """
        session_key = f"{user_id}_{conversation_id}"
        
        # Check if session exists
        if session_key not in self.sessions:
            return False
        
        try:
            # Get session and clean up
            session = self.sessions[session_key]
            await session.cleanup()
            
            # Remove from tracking
            del self.sessions[session_key]
            
            # Update stats
            self.stats["total_sessions_archived"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return False
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current sessions.
        
        Returns:
            Session statistics
        """
        # Count sessions by user
        user_counts = {}
        for session_key in self.sessions:
            user_id = int(session_key.split('_')[0])
            if user_id not in user_counts:
                user_counts[user_id] = 0
            user_counts[user_id] += 1
        
        # Get resource stats
        resource_stats = await self._get_resource_stats()
        
        # Get central brain stats
        brain_stats = {}
        for user_conv, brain in self.central_brain_instances.items():
            try:
                stats = await brain.get_system_stats()
                brain_stats[user_conv] = {
                    "memory_operations": stats.get("memory_stats", {}).get("total_operations", 0),
                    "emotional_updates": stats.get("performance_metrics", {}).get("emotion_updates", 0),
                    "avg_response_time": stats.get("performance_metrics", {}).get("avg_response_time", 0)
                }
            except Exception as e:
                logger.warning(f"Failed to get brain stats for {user_conv}: {e}")
        
        return {
            "active_sessions": len(self.sessions),
            "sessions_per_user": user_counts,
            "session_stats": self.stats,
            "resource_stats": resource_stats,
            "brain_stats": brain_stats
        }
    
    async def get_or_create_central_brain(
        self, 
        user_id: int, 
        conversation_id: int
    ) -> NyxBrain:
        """
        Get or create a central brain instance.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            NyxBrain instance
        """
        key = f"{user_id}_{conversation_id}"
        
        if key not in self.central_brain_instances:
            brain = await NyxBrain.get_instance(user_id, conversation_id)
            self.central_brain_instances[key] = brain
        
        return self.central_brain_instances[key]
    
    def _check_session_limits(self, user_id: int) -> bool:
        """Check if session limits are reached."""
        # Count user's sessions
        user_sessions = sum(
            1 for key in self.sessions 
            if key.startswith(f"{user_id}_")
        )
        
        # Check per-user limit
        if user_sessions >= self.max_sessions_per_user:
            return False
        
        # Check total limit
        if len(self.sessions) >= self.max_total_sessions:
            return False
        
        return True
    
    def _update_concurrent_stats(self) -> None:
        """Update concurrent session stats."""
        current_count = len(self.sessions)
        if current_count > self.stats["peak_concurrent_sessions"]:
            self.stats["peak_concurrent_sessions"] = current_count
    
    def _ensure_maintenance_running(self) -> None:
        """Ensure maintenance task is running."""
        if self.maintenance_task is None or self.maintenance_task.done():
            self.running = True
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def _maintenance_loop(self) -> None:
        """Maintenance loop for session cleanup and monitoring."""
        try:
            while self.running and self.sessions:
                # Check for inactive sessions
                await self._cleanup_inactive_sessions()
                
                # Update resource stats
                await self._update_resource_stats()
                
                # Wait before next check
                await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            logger.error(f"Error in maintenance loop: {e}")
        finally:
            self.running = False
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Clean up inactive sessions."""
        current_time = time.time()
        to_close = []
        
        # Find sessions to close
        for session_key, session in self.sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                to_close.append(session_key)
        
        # Close sessions
        for session_key in to_close:
            user_id, conversation_id = map(int, session_key.split('_'))
            logger.info(f"Auto-closing inactive session: {session_key}")
            await self.close_session(user_id, conversation_id)
    
    async def _update_resource_stats(self) -> None:
        """Update resource usage statistics."""
        import psutil
        
        # Global process stats
        process = psutil.Process()
        self.resource_usage["total_memory"] = process.memory_info().rss
        self.resource_usage["total_cpu"] = process.cpu_percent()
        
        # Per-session stats (simplified version)
        for session_key, session in self.sessions.items():
            self.resource_usage["sessions"][session_key] = {
                "last_activity": session.last_activity,
                "turns": session.turn_count
            }
    
    async def _get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        # Calculate average activity time
        avg_activity = 0
        if self.sessions:
            current_time = time.time()
            total_idle = sum(
                current_time - session.last_activity
                for session in self.sessions.values()
            )
            avg_activity = total_idle / len(self.sessions)
        
        return {
            "memory_usage": self.resource_usage.get("total_memory", 0),
            "cpu_usage": self.resource_usage.get("total_cpu", 0),
            "avg_session_idle_time": avg_activity,
            "session_count": len(self.sessions)
        }
