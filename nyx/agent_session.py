# nyx/agent_session.py

import asyncio
import logging
import time
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple, Union

import aiohttp

from nyx.nyx_agent_sdk import process_user_input
from nyx.nyx_governance import NyxUnifiedGovernor

logger = logging.getLogger(__name__)

class NyxAgentSession:
    """
    Lightweight wrapper for a single agent session.
    
    This class connects an individual agent instance to the central 
    NyxBrain, handling local state, learning reporting, and resource
    management.
    """
    
    def __init__(
        self, 
        user_id: int, 
        conversation_id: int, 
        central_brain_url: Optional[str] = None,
        direct_brain_access: bool = True
    ):
        """
        Initialize a new agent session.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            central_brain_url: URL for central brain API (optional)
            direct_brain_access: Whether to use direct Python access to brain
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.central_brain_url = central_brain_url
        self.direct_brain_access = direct_brain_access
        
        # Session state
        self.session_id = None
        self.emotional_state = {}
        self.context_cache = {}
        self.turn_count = 0
        self.last_activity = time.time()
        
        # Learning collection
        self.learning_queue = []
        self.flush_interval = 5  # Flush learning every N turns
        self.last_flush_time = time.time()
        
        # Background tasks
        self.background_tasks = []
        
        # Governance
        self.governance = None
        
        # Initialized flag
        self.initialized = False
    
    async def initialize(self, initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Initialize the agent session and register with central brain.
        
        Args:
            initial_context: Optional initial context data
            
        Returns:
            session_id: Unique session identifier
        """
        if self.initialized:
            return self.session_id
        
        # Initialize governance
        self.governance = NyxUnifiedGovernor(self.user_id, self.conversation_id)
        
        # Register with central brain
        self.session_id = await self._register_with_brain(initial_context)
        
        # Start background tasks
        self._start_background_tasks()
        
        self.initialized = True
        logger.info(f"Agent session initialized with ID: {self.session_id}")
        
        return self.session_id
    
    async def process_user_input(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input and generate a response.
        
        Args:
            user_input: User's message text
            context: Additional context data
            
        Returns:
            Response data
        """
        if not self.initialized:
            await self.initialize()
        
        self.turn_count += 1
        self.last_activity = time.time()
        
        # Prepare context
        enhanced_context = self._enhance_context(context)
        
        # Process input using agent SDK
        result = await process_user_input(
            self.user_id,
            self.conversation_id,
            user_input,
            enhanced_context
        )
        
        # Cache context updates
        if "context_updates" in result:
            self._update_context_cache(result["context_updates"])
        
        # Update emotional state if provided
        if "emotional_state" in result:
            self.emotional_state = result["emotional_state"]
            
            # Check for emotional spike
            if self._is_emotional_spike(result["emotional_state"]):
                await self._record_emotional_learning(user_input, result)
        
        # Check for user revelations
        if "user_revelations" in result:
            await self._record_user_revelation(user_input, result)
        
        # Record interaction learning
        await self._record_interaction_learning(user_input, result)
        
        # Flush learning queue if needed
        if self.turn_count % self.flush_interval == 0:
            await self._flush_learning_queue()
        
        # Update session data
        await self._update_session_data(user_input, result)
        
        # Check for critical actions that need validation
        if self._is_critical_action(result):
            action_data = {
                "user_input": user_input,
                "result": result,
                "context": context
            }
            
            # Validate action with brain
            is_valid = await self.validate_action("response_generation", action_data)
            
            if not is_valid:
                # Action rejected by brain, generate safe fallback
                result = await self._generate_safe_fallback(user_input, context)
        
        return result
    
    def _is_critical_action(self, result: Dict[str, Any]) -> bool:
        """Determine if an action is critical and needs brain validation."""
        # Check for potential risky actions
        if "response_type" in result and result["response_type"] in ["critical", "sensitive", "high_risk"]:
            return True
        
        # Check for content sensitivity
        if "content_sensitivity" in result and result["content_sensitivity"] >= 0.7:
            return True
        
        # Check for user model changes
        if "user_model_updates" in result and result["user_model_updates"]:
            return True
            
        return False
    
    async def _generate_safe_fallback(self, user_input: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a safe fallback when a critical action is rejected."""
        # Generate a non-committal, safe response
        safe_result = {
            "message": "I'd like to better understand your request before proceeding. Could you provide more context?",
            "response_type": "clarification",
            "emotional_state": self.emotional_state
        }
        
        return safe_result
    
    async def cleanup(self) -> Dict[str, Any]:
        """
        Clean up resources and archive the session.
        
        Returns:
            Cleanup status
        """
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Flush any remaining learning
        await self._flush_learning_queue()
        
        # Archive session with central brain
        if self.session_id:
            try:
                archive_result = await self._brain_request(
                    "archive_session",
                    {"session_id": self.session_id}
                )
                
                return {
                    "success": True,
                    "archive_status": archive_result
                }
            except Exception as e:
                logger.error(f"Error archiving session: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        return {"success": False, "error": "Session not initialized"}
    
    async def _register_with_brain(
        self, 
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register this session with the central brain."""
        try:
            # Prepare request data
            request_data = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "initial_context": initial_context or {}
            }
            
            # Make request to central brain
            response = await self._brain_request("register_session", request_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to register with central brain: {e}")
            # Generate local fallback session ID
            return f"local_session_{self.user_id}_{self.conversation_id}_{int(time.time())}"
    
    def _enhance_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance context with cached data."""
        # Start with provided context
        enhanced = context.copy() if context else {}
        
        # Add cached context
        for key, value in self.context_cache.items():
            if key not in enhanced:
                enhanced[key] = value
        
        # Add session information
        enhanced["session_id"] = self.session_id
        enhanced["turn_count"] = self.turn_count
        
        # Add emotional state if available
        if self.emotional_state:
            enhanced["emotional_state"] = self.emotional_state
        
        return enhanced
    
    def _update_context_cache(self, updates: Dict[str, Any]) -> None:
        """Update context cache with new values."""
        for key, value in updates.items():
            self.context_cache[key] = value
    
    def _is_emotional_spike(self, current_state: Dict[str, Any]) -> bool:
        """Determine if there was a significant emotional change."""
        if not self.emotional_state:
            return False
        
        # Check for significant changes in major emotions
        for emotion in ["valence", "arousal", "dominance"]:
            if emotion in current_state and emotion in self.emotional_state:
                change = abs(current_state[emotion] - self.emotional_state[emotion])
                if change > 0.3:  # Significant change threshold
                    return True
        
        # Check for new dominant emotion
        if "dominant_emotion" in current_state and "dominant_emotion" in self.emotional_state:
            if current_state["dominant_emotion"] != self.emotional_state["dominant_emotion"]:
                return True
        
        return False
    
    async def _record_emotional_learning(
        self, 
        user_input: str, 
        result: Dict[str, Any]
    ) -> None:
        """Record emotional spike learning."""
        self.learning_queue.append({
            "trigger": "emotional_spike",
            "tags": ["emotional", "state_change"],
            "content": f"Emotional state shifted significantly during interaction",
            "source": "agent_session",
            "emotional_snapshot": result["emotional_state"],
            "user_input": user_input,
            "response": result.get("message", "")
        })
    
    async def _record_user_revelation(
        self, 
        user_input: str, 
        result: Dict[str, Any]
    ) -> None:
        """Record user revelation learning."""
        revelations = result["user_revelations"]
        
        self.learning_queue.append({
            "trigger": "user_reveal",
            "tags": ["revelation", "user_preference"],
            "content": f"User revealed new information or preferences",
            "source": "agent_session",
            "revelations": revelations,
            "user_input": user_input
        })
    
    async def _record_interaction_learning(
        self, 
        user_input: str, 
        result: Dict[str, Any]
    ) -> None:
        """Record general interaction learning."""
        self.learning_queue.append({
            "trigger": "time_checkpoint",
            "tags": ["interaction", "message"],
            "content": f"Interaction processed - Turn #{self.turn_count}",
            "source": "agent_session",
            "interaction_type": "user_message",
            "user_input": user_input[:100] + ("..." if len(user_input) > 100 else ""),
            "response_type": result.get("response_type", "standard")
        })
    
    async def _flush_learning_queue(self) -> None:
        """Flush queued learning to the central brain."""
        if not self.learning_queue:
            return
        
        try:
            # Group by trigger type to reduce requests
            triggers = {}
            for item in self.learning_queue:
                trigger = item["trigger"]
                if trigger not in triggers:
                    triggers[trigger] = []
                triggers[trigger].append(item)
            
            # Send each group
            for trigger, items in triggers.items():
                if len(items) == 1:
                    # Send single item
                    await self._brain_request(
                        "report_learning",
                        {
                            "session_id": self.session_id,
                            "learning": items[0]
                        }
                    )
                else:
                    # Send batch
                    batch = {
                        "trigger": trigger,
                        "tags": ["batch", f"{trigger}_batch"],
                        "content": f"Batch of {len(items)} {trigger} events",
                        "source": "agent_session",
                        "batch_size": len(items),
                        "batch_items": items
                    }
                    
                    await self._brain_request(
                        "report_learning",
                        {
                            "session_id": self.session_id,
                            "learning": batch
                        }
                    )
            
            # Clear queue and update flush time
            self.learning_queue = []
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing learning queue: {e}")
    
    async def _update_session_data(
        self, 
        user_input: str, 
        result: Dict[str, Any]
    ) -> None:
        """Update session data with the central brain."""
        try:
            # Prepare update data
            update_data = {
                "last_active": datetime.datetime.now().isoformat(),
                "current_context": {
                    "last_user_input": user_input[:100] + ("..." if len(user_input) > 100 else ""),
                    "last_response_type": result.get("response_type", "standard")
                }
            }
            
            # Add emotional state if available
            if "emotional_state" in result:
                update_data["emotional_state"] = result["emotional_state"]
            
            # Make request to central brain
            await self._brain_request(
                "update_session",
                {
                    "session_id": self.session_id,
                    "update_data": update_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating session data: {e}")
    
    async def _brain_request(self, method: str, data: Dict[str, Any]) -> Any:
        """Make a request to the central brain."""
        if self.direct_brain_access:
            # Direct Python access to brain
            try:
                from nyx.nyx_brain import NyxBrain
                brain = await NyxBrain.get_instance(self.user_id, self.conversation_id)
                
                # Call the appropriate method
                if not hasattr(brain, method):
                    raise ValueError(f"Brain has no method: {method}")
                
                method_func = getattr(brain, method)
                return await method_func(**data)
                
            except Exception as e:
                logger.error(f"Error in direct brain access: {e}")
                raise
                
        elif self.central_brain_url:
            # Remote API access
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.central_brain_url}/{method}",
                        json=data
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise ValueError(f"Brain API error: {response.status} - {error_text}")
                            
            except Exception as e:
                logger.error(f"Error in brain API request: {e}")
                raise
                
        else:
            raise ValueError("No brain access method configured")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for session maintenance."""
        # Periodic learning flush
        self.background_tasks.append(
            asyncio.create_task(self._periodic_learning_flush())
        )
        
        # Periodic heartbeat
        self.background_tasks.append(
            asyncio.create_task(self._periodic_heartbeat())
        )
    
    async def _periodic_learning_flush(self) -> None:
        """Periodically flush learning queue."""
        while True:
            await asyncio.sleep(300)  # 5 minutes
            
            # Check if flush is needed
            if self.learning_queue and time.time() - self.last_flush_time > 300:
                await self._flush_learning_queue()
    
    async def _periodic_heartbeat(self) -> None:
        """Send periodic heartbeats to keep session active."""
        while True:
            await asyncio.sleep(60)  # 1 minute
            
            try:
                # Skip if there was recent activity
                if time.time() - self.last_activity < 60:
                    continue
                
                # Update last active time
                await self._brain_request(
                    "update_session",
                    {
                        "session_id": self.session_id,
                        "update_data": {
                            "last_active": datetime.datetime.now().isoformat()
                        }
                    }
                )
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
    
    async def validate_action(self, action_type: str, action_data: Dict[str, Any]) -> bool:
        """Validate critical actions with the brain before execution."""
        try:
            # Get validation from brain
            validation_result = await self._brain_request(
                "validate_action",
                {
                    "session_id": self.session_id,
                    "action_type": action_type,
                    "action_data": action_data,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
            
            return validation_result.get("valid", False)
        except Exception as e:
            logger.error(f"Error validating action with brain: {e}")
            # Default to rejecting the action if validation fails
            return False
    
    # Modify process_user_input to validate critical actions
    async def process_user_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Existing code...
        
        # Check for critical actions that need validation
        if self._is_critical_action(result):
            action_data = {
                "user_input": user_input,
                "result": result,
                "context": context
            }
            
            # Validate action with brain
            is_valid = await self.validate_action("response_generation", action_data)
            
            if not is_valid:
                # Action rejected by brain, generate safe fallback
                result = await self._generate_safe_fallback(user_input, context)
        
        return result
    
    def _is_critical_action(self, result: Dict[str, Any]) -> bool:
        """Determine if an action is critical and needs brain validation."""
        # Check for potential risky actions
        if "response_type" in result and result["response_type"] in ["critical", "sensitive", "high_risk"]:
            return True
        
        # Check for content sensitivity
        if "content_sensitivity" in result and result["content_sensitivity"] >= 0.7:
            return True
        
        # Check for user model changes
        if "user_model_updates" in result and result["user_model_updates"]:
            return True
            
        return False
    
    async def _generate_safe_fallback(self, user_input: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a safe fallback when a critical action is rejected."""
        # Generate a non-committal, safe response
        safe_result = {
            "message": "I'd like to better understand your request before proceeding. Could you provide more context?",
            "response_type": "clarification",
            "emotional_state": self.emotional_state
        }
        
        return safe_result
        
     async def validate_action(self, action_type: str, action_data: Dict[str, Any]) -> bool:
            """Validate critical actions with the brain before execution."""
            try:
                # Get validation from brain
                validation_result = await self._brain_request(
                    "validate_action",
                    {
                        "session_id": self.session_id,
                        "action_type": action_type,
                        "action_data": action_data,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                
                return validation_result.get("valid", False)
            except Exception as e:
                logger.error(f"Error validating action with brain: {e}")
                # Default to rejecting the action if validation fails
                return False
