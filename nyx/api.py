# nyx/api.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional

from nyx.session_factory import NyxSessionFactory
from nyx.resource_monitor import ResourceMonitor
from nyx.nyx_brain import NyxBrain

logger = logging.getLogger(__name__)

class NyxAPI:
    """
    API for interacting with Nyx session-based system.
    
    This provides a unified interface for all client applications
    to interact with the session-based architecture.
    """
    
    def __init__(self):
        """Initialize the API."""
        # Get session factory instance
        self.session_factory = NyxSessionFactory.get_instance()
        
        # Get resource monitor instance
        self.resource_monitor = ResourceMonitor.get_instance()
        self.resource_monitor.start_monitoring()
        
        # Track session references
        self.last_session_access = {}
    
    async def process_message(
        self, 
        user_id: int, 
        conversation_id: int, 
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            message: User message
            context: Optional additional context
            
        Returns:
            Response data
        """
        # Get or create session
        session = await self.session_factory.get_session(
            user_id, 
            conversation_id
        )
        
        if not session:
            return {
                "success": False,
                "error": "Failed to create session"
            }
        
        # Track access
        self.last_session_access[f"{user_id}_{conversation_id}"] = time.time()
        
        # Process message
        try:
            # Enhance context with resource data
            enhanced_context = context.copy() if context else {}
            
            # Process message
            result = await session.process_user_input(message, enhanced_context)
            
            return {
                "success": True,
                "response": result
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close_conversation(self, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Close a conversation and cleanup resources.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Status information
        """
        # Remove from access tracking
        session_key = f"{user_id}_{conversation_id}"
        if session_key in self.last_session_access:
            del self.last_session_access[session_key]
        
        # Close session
        success = await self.session_factory.close_session(user_id, conversation_id)
        
        if success:
            return {
                "success": True,
                "status": "conversation_closed"
            }
        else:
            return {
                "success": False,
                "error": "Failed to close conversation"
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information.
        
        Returns:
            System status data
        """
        # Get session stats
        session_stats = await self.session_factory.get_session_stats()
        
        # Get health metrics
        health_metrics = await self.resource_monitor.get_health_metrics()
        
        # Combine information
        return {
            "success": True,
            "session_stats": session_stats,
            "health_metrics": health_metrics,
            "active_conversations": len(self.last_session_access),
            "status": "operational" if health_metrics["system_health"] > 0.5 else "degraded"
        }
    
    async def get_conversation_insights(
        self, 
        user_id: int, 
        conversation_id: int
    ) -> Dict[str, Any]:
        """
        Get insights for a specific conversation.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Conversation insights
        """
        # Get central brain reference
        try:
            brain = await self.session_factory.get_or_create_central_brain(
                user_id, 
                conversation_id
            )
            
            # Get stats from central brain
            stats = await brain.get_system_stats()
            
            # Get active session data
            session = await self.session_factory.get_session(
                user_id, 
                conversation_id,
                create_if_missing=False
            )
            
            session_data = {}
            if session:
                session_data = {
                    "active": True,
                    "session_id": session.session_id,
                    "turn_count": session.turn_count,
                    "last_activity": session.last_activity,
                    "emotional_state": session.emotional_state
                }
            else:
                session_data = {
                    "active": False
                }
            
            return {
                "success": True,
                "brain_stats": stats,
                "session_data": session_data
            }
        except Exception as e:
            logger.error(f"Error getting conversation insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_cross_user_insights(self) -> Dict[str, Any]:
        """
        Get insights across all users and conversations.
        
        Returns:
            Cross-user insights
        """
        try:
            # Create a temporary brain for global insights
            brain = await NyxBrain.get_instance(0, 0)  # System brain
            
            # Get global memory stats
            memory_stats = await brain.memory_core.get_global_memory_stats()
            
            # Get learning patterns
            learning_patterns = await brain.get_learning_patterns()
            
            # Get sentiment trends
            sentiment_trends = await brain.get_sentiment_trends()
            
            return {
                "success": True,
                "memory_stats": memory_stats,
                "learning_patterns": learning_patterns,
                "sentiment_trends": sentiment_trends,
                "active_sessions": len(self.session_factory.sessions)
            }
        except Exception as e:
            logger.error(f"Error getting cross-user insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def set_user_priority(
        self, 
        user_id: int, 
        priority: str
    ) -> Dict[str, Any]:
        """
        Set priority for a user.
        
        Args:
            user_id: User ID
            priority: Priority level ('high', 'medium', 'low', 'background')
            
        Returns:
            Status information
        """
        # Set priority
        success = await self.resource_monitor.set_user_priority(user_id, priority)
        
        if success:
            return {
                "success": True,
                "user_id": user_id,
                "priority": priority
            }
        else:
            return {
                "success": False,
                "error": f"Invalid priority: {priority}"
            }
    
    async def cleanup_inactive_sessions(self, max_idle_time: int = 3600) -> Dict[str, Any]:
        """
        Cleanup inactive sessions.
        
        Args:
            max_idle_time: Maximum idle time in seconds
            
        Returns:
            Cleanup results
        """
        try:
            # Find inactive sessions
            current_time = time.time()
            inactive_sessions = []
            
            for session_key, last_access in self.last_session_access.items():
                if current_time - last_access > max_idle_time:
                    inactive_sessions.append(session_key)
            
            # Close inactive sessions
            closed_sessions = []
            for session_key in inactive_sessions:
                user_id, conversation_id = map(int, session_key.split('_'))
                
                # Close session
                success = await self.session_factory.close_session(user_id, conversation_id)
                
                if success:
                    closed_sessions.append(session_key)
                    
                    # Remove from access tracking
                    if session_key in self.last_session_access:
                        del self.last_session_access[session_key]
            
            return {
                "success": True,
                "inactive_sessions": len(inactive_sessions),
                "closed_sessions": len(closed_sessions)
            }
        except Exception as e:
            logger.error(f"Error cleaning up inactive sessions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Create global instance
nyx_api = NyxAPI()

# Convenience functions for API access

async def process_message(
    user_id: int, 
    conversation_id: int, 
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process user message through Nyx API."""
    return await nyx_api.process_message(user_id, conversation_id, message, context)

async def close_conversation(
    user_id: int, 
    conversation_id: int
) -> Dict[str, Any]:
    """Close conversation through Nyx API."""
    return await nyx_api.close_conversation(user_id, conversation_id)

async def get_system_status() -> Dict[str, Any]:
    """Get system status through Nyx API."""
    return await nyx_api.get_system_status()

async def get_conversation_insights(
    user_id: int, 
    conversation_id: int
) -> Dict[str, Any]:
    """Get conversation insights through Nyx API."""
    return await nyx_api.get_conversation_insights(user_id, conversation_id)

async def get_cross_user_insights() -> Dict[str, Any]:
    """Get cross-user insights through Nyx API."""
    return await nyx_api.get_cross_user_insights()

async def set_user_priority(
    user_id: int, 
    priority: str
) -> Dict[str, Any]:
    """Set user priority through Nyx API."""
    return await nyx_api.set_user_priority(user_id, priority)

async def cleanup_inactive_sessions(
    max_idle_time: int = 3600
) -> Dict[str, Any]:
    """Cleanup inactive sessions through Nyx API."""
    return await nyx_api.cleanup_inactive_sessions(max_idle_time)
