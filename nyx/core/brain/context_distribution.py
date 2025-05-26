# nyx/core/brain/context_distribution.py

import asyncio
import logging
import datetime
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ContextScope(Enum):
    """Defines the scope of context sharing"""
    GLOBAL = auto()      # Available to all modules
    TARGETED = auto()    # Available to specific modules
    PRIVATE = auto()     # Module-specific context
    SHARED = auto()      # Shared between related modules

class ContextPriority(Enum):
    """Priority levels for context updates"""
    CRITICAL = auto()    # Must be processed immediately
    HIGH = auto()        # High priority processing
    NORMAL = auto()      # Standard priority
    MEDIUM = NORMAL
    LOW = auto()         # Can be deferred

@dataclass
class ContextUpdate:
    """Represents a context update from a module"""
    source_module: str
    update_type: str
    data: Dict[str, Any]
    scope: ContextScope = ContextScope.GLOBAL
    priority: ContextPriority = ContextPriority.NORMAL
    target_modules: Optional[List[str]] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    ttl_seconds: Optional[int] = None  # Time to live

class SharedContext(BaseModel):
    """Central context object shared across all modules"""
    
    # Core context
    user_input: str = ""
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Active processing context
    active_modules: Set[str] = Field(default_factory=set)
    processing_stage: str = "input"  # input, processing, synthesis, output
    task_purpose: str = "communicate"
    
    # Module states - each module can contribute its state
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    memory_context: Dict[str, Any] = Field(default_factory=dict)
    goal_context: Dict[str, Any] = Field(default_factory=dict)
    relationship_context: Dict[str, Any] = Field(default_factory=dict)
    attention_context: Dict[str, Any] = Field(default_factory=dict)
    mode_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Cross-module communications
    module_messages: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    context_updates: List[ContextUpdate] = Field(default_factory=list)
    
    # Processing results
    module_outputs: Dict[str, Any] = Field(default_factory=dict)
    synthesis_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class ContextDistributionSystem:
    """
    Manages context distribution and module coordination for NyxBrain
    Implements a2a principles for cohesive module integration
    """
    
    def __init__(self, nyx_brain):
        self.nyx_brain = nyx_brain
        self.current_context: Optional[SharedContext] = None
        self.module_subscriptions: Dict[str, Set[str]] = {}  # module -> context_types
        self.context_history: List[SharedContext] = []
        self.max_history = 10
        
        # Processing coordination
        self.processing_lock = asyncio.Lock()
        self.context_update_queue = asyncio.Queue()
        self.active_processing = False
        
        logger.info("ContextDistributionSystem initialized")
    
    async def initialize_context_session(self, 
                                       user_input: str, 
                                       user_id: Optional[str] = None,
                                       initial_context: Dict[str, Any] = None) -> SharedContext:
        """
        Initialize a new context session for processing input
        
        Args:
            user_input: The user's input
            user_id: User identifier
            initial_context: Additional initial context
            
        Returns:
            Initialized SharedContext
        """
        async with self.processing_lock:
            # Create new shared context
            self.current_context = SharedContext(
                user_input=user_input,
                user_id=user_id,
                conversation_id=str(getattr(self.nyx_brain, 'conversation_id', 'unknown')),
                session_context=initial_context or {}
            )
            
            # Determine active modules
            active_modules = await self.nyx_brain._determine_active_modules(
                self.current_context.session_context, user_input
            )
            self.current_context.active_modules = active_modules
            
            # Broadcast initial context to all active modules
            await self._broadcast_context_initialization()
            
            logger.debug(f"Initialized context session with {len(active_modules)} active modules")
            return self.current_context
    
    async def _broadcast_context_initialization(self):
        """Broadcast the initial context to all active modules"""
        if not self.current_context:
            return
            
        initialization_tasks = []
        
        for module_name in self.current_context.active_modules:
            if hasattr(self.nyx_brain, module_name):
                module = getattr(self.nyx_brain, module_name)
                if hasattr(module, 'receive_context'):
                    task = self._safe_module_context_init(module_name, module)
                    initialization_tasks.append(task)
        
        # Wait for all modules to receive initial context
        if initialization_tasks:
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
    
    async def _safe_module_context_init(self, module_name: str, module):
        """Safely initialize context for a module"""
        try:
            await module.receive_context(self.current_context)
            logger.debug(f"Context initialized for module: {module_name}")
        except Exception as e:
            logger.error(f"Error initializing context for {module_name}: {e}")
    
    async def coordinate_processing_stage(self, stage: str) -> Dict[str, Any]:
        """
        Coordinate a processing stage across all active modules
        
        Args:
            stage: The processing stage name
            
        Returns:
            Results from all modules for this stage
        """
        if not self.current_context:
            raise ValueError("No active context session")
        
        self.current_context.processing_stage = stage
        self.current_context.last_updated = datetime.datetime.now()
        
        stage_results = {}
        processing_tasks = []
        
        for module_name in self.current_context.active_modules:
            if hasattr(self.nyx_brain, module_name):
                module = getattr(self.nyx_brain, module_name)
                
                # Check if module has a handler for this stage
                stage_handler = getattr(module, f'process_{stage}', None)
                if stage_handler and callable(stage_handler):
                    task = self._safe_module_stage_processing(
                        module_name, module, stage_handler, stage
                    )
                    processing_tasks.append(task)
        
        # Process all modules for this stage
        if processing_tasks:
            task_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Collect results
            for i, result in enumerate(task_results):
                module_name = list(self.current_context.active_modules)[i] if i < len(self.current_context.active_modules) else f"module_{i}"
                if isinstance(result, Exception):
                    logger.error(f"Error in {module_name} during {stage}: {result}")
                    stage_results[module_name] = {"error": str(result)}
                else:
                    stage_results[module_name] = result
        
        # Store results in context
        self.current_context.module_outputs[stage] = stage_results
        
        # Process any context updates that were generated
        await self._process_pending_context_updates()
        
        return stage_results
    
    async def _safe_module_stage_processing(self, module_name: str, module, handler, stage: str):
        """Safely process a stage for a module"""
        try:
            result = await handler(self.current_context)
            logger.debug(f"Completed {stage} processing for {module_name}")
            return result
        except Exception as e:
            logger.error(f"Error in {module_name} during {stage}: {e}")
            return {"error": str(e)}
    
    async def add_context_update(self, update: ContextUpdate):
        """Add a context update to the processing queue"""
        await self.context_update_queue.put(update)
        
        # Process immediately if critical priority
        if update.priority == ContextPriority.CRITICAL:
            await self._process_pending_context_updates()
    
    async def _process_pending_context_updates(self):
        """Process all pending context updates"""
        while not self.context_update_queue.empty():
            try:
                update = await asyncio.wait_for(self.context_update_queue.get(), timeout=0.1)
                await self._apply_context_update(update)
                self.context_update_queue.task_done()
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing context update: {e}")
    
    async def _apply_context_update(self, update: ContextUpdate):
        """Apply a context update to the shared context"""
        if not self.current_context:
            return
        
        # Check TTL
        if update.ttl_seconds:
            age = (datetime.datetime.now() - update.timestamp).total_seconds()
            if age > update.ttl_seconds:
                logger.debug(f"Context update from {update.source_module} expired")
                return
        
        # Apply update based on scope
        if update.scope == ContextScope.GLOBAL:
            # Update global context
            self._merge_update_data(update.data)
            
        elif update.scope == ContextScope.TARGETED and update.target_modules:
            # Send to specific modules
            await self._send_targeted_update(update)
            
        elif update.scope == ContextScope.SHARED:
            # Add to module messages for related modules
            self._add_to_module_messages(update)
        
        # Store update in context history
        self.current_context.context_updates.append(update)
        self.current_context.last_updated = datetime.datetime.now()
    
    def _merge_update_data(self, data: Dict[str, Any]):
        """Merge update data into the current context"""
        for key, value in data.items():
            if key in ['emotional_state', 'memory_context', 'goal_context', 
                      'relationship_context', 'attention_context', 'mode_context']:
                # Merge context dictionaries
                current_dict = getattr(self.current_context, key, {})
                if isinstance(current_dict, dict) and isinstance(value, dict):
                    current_dict.update(value)
                else:
                    setattr(self.current_context, key, value)
            else:
                # Direct assignment for other fields
                if hasattr(self.current_context, key):
                    setattr(self.current_context, key, value)
                else:
                    # Add to session context if not a known field
                    self.current_context.session_context[key] = value
    
    async def _send_targeted_update(self, update: ContextUpdate):
        """Send targeted update to specific modules"""
        for module_name in update.target_modules or []:
            if (module_name in self.current_context.active_modules and 
                hasattr(self.nyx_brain, module_name)):
                
                module = getattr(self.nyx_brain, module_name)
                if hasattr(module, 'receive_context_update'):
                    try:
                        await module.receive_context_update(update)
                    except Exception as e:
                        logger.error(f"Error sending update to {module_name}: {e}")
    
    def _add_to_module_messages(self, update: ContextUpdate):
        """Add update to module messages"""
        source = update.source_module
        if source not in self.current_context.module_messages:
            self.current_context.module_messages[source] = []
        
        self.current_context.module_messages[source].append({
            'type': update.update_type,
            'data': update.data,
            'timestamp': update.timestamp.isoformat(),
            'priority': update.priority.name
        })
    
    async def synthesize_responses(self) -> Dict[str, Any]:
        """
        Synthesize responses from all active modules into a cohesive output
        
        Returns:
            Synthesized response data
        """
        if not self.current_context:
            raise ValueError("No active context session")
        
        self.current_context.processing_stage = "synthesis"
        
        # Gather all module outputs
        all_outputs = {}
        for stage, stage_outputs in self.current_context.module_outputs.items():
            all_outputs[stage] = stage_outputs
        
        # Run synthesis coordination
        synthesis_result = await self.coordinate_processing_stage("synthesis")
        
        # Create final synthesis
        final_synthesis = {
            "primary_response": self._extract_primary_response(all_outputs),
            "emotional_context": self.current_context.emotional_state,
            "memory_integration": self.current_context.memory_context,
            "goal_alignment": self.current_context.goal_context,
            "relationship_considerations": self.current_context.relationship_context,
            "attention_allocation": self.current_context.attention_context,
            "mode_adjustments": self.current_context.mode_context,
            "module_contributions": all_outputs,
            "synthesis_metadata": {
                "active_modules": list(self.current_context.active_modules),
                "processing_stages": list(self.current_context.module_outputs.keys()),
                "context_updates": len(self.current_context.context_updates),
                "session_duration": (datetime.datetime.now() - self.current_context.created_at).total_seconds()
            }
        }
        
        self.current_context.synthesis_results = final_synthesis
        return final_synthesis
    
    def _extract_primary_response(self, all_outputs: Dict[str, Any]) -> str:
        """Extract the primary response from module outputs"""
        # Priority order for response sources
        response_priorities = [
            'agentic_action_generator',
            'proactive_communication_engine', 
            'reasoning_core',
            'emotional_core'
        ]
        
        for source in response_priorities:
            for stage_outputs in all_outputs.values():
                if source in stage_outputs:
                    output = stage_outputs[source]
                    if isinstance(output, dict):
                        for response_key in ['response', 'message', 'text', 'output']:
                            if response_key in output:
                                return str(output[response_key])
                    elif isinstance(output, str):
                        return output
        
        # Fallback to any string response
        for stage_outputs in all_outputs.values():
            for module_output in stage_outputs.values():
                if isinstance(module_output, str) and len(module_output) > 10:
                    return module_output
                elif isinstance(module_output, dict):
                    for key, value in module_output.items():
                        if isinstance(value, str) and len(value) > 10:
                            return value
        
        return "I'm processing your input across multiple systems."
    
    async def finalize_context_session(self):
        """Finalize the current context session"""
        if not self.current_context:
            return
        
        # Add to history
        self.context_history.append(self.current_context)
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]
        
        # Notify modules of session end
        for module_name in self.current_context.active_modules:
            if hasattr(self.nyx_brain, module_name):
                module = getattr(self.nyx_brain, module_name)
                if hasattr(module, 'finalize_context_session'):
                    try:
                        await module.finalize_context_session(self.current_context)
                    except Exception as e:
                        logger.error(f"Error finalizing session for {module_name}: {e}")
        
        # Clear current context
        self.current_context = None
        logger.debug("Context session finalized")
    
    def get_context_for_module(self, module_name: str) -> Optional[SharedContext]:
        """Get the current context for a specific module"""
        if (self.current_context and 
            module_name in self.current_context.active_modules):
            return self.current_context
        return None
    
    def subscribe_module_to_context(self, module_name: str, context_types: List[str]):
        """Subscribe a module to specific context update types"""
        if module_name not in self.module_subscriptions:
            self.module_subscriptions[module_name] = set()
        self.module_subscriptions[module_name].update(context_types)
    
    async def get_cross_module_communication(self, requesting_module: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get messages from other modules for the requesting module"""
        if not self.current_context:
            return {}
        
        # Return messages from all other active modules
        relevant_messages = {}
        for module_name, messages in self.current_context.module_messages.items():
            if module_name != requesting_module:
                relevant_messages[module_name] = messages
        
        return relevant_messages
