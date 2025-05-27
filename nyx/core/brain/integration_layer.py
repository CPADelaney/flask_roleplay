# nyx/core/brain/integration_layer.py

import asyncio
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from .context_distribution import (
    ContextDistributionSystem, SharedContext, ContextUpdate, 
    ContextScope, ContextPriority
)

logger = logging.getLogger(__name__)

class ContextAwareModule(ABC):
    """
    Base class for modules that participate in context distribution
    Modules should inherit from this to enable a2a coordination
    """
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.current_context: Optional[SharedContext] = None
        self.context_subscriptions: List[str] = []
        
    async def receive_context(self, context: SharedContext):
        """
        Receive initial context for a processing session
        
        Args:
            context: The shared context object
        """
        self.current_context = context
        await self.on_context_received(context)
    
    async def on_context_received(self, context: SharedContext):
        """
        Override this method to handle context initialization
        
        Args:
            context: The shared context object
        """
        pass
    
    async def receive_context_update(self, update: ContextUpdate):
        """
        Receive a context update from another module
        
        Args:
            update: The context update
        """
        await self.on_context_update(update)
    
    async def on_context_update(self, update: ContextUpdate):
        """
        Override this method to handle context updates
        
        Args:
            update: The context update
        """
        pass
    
    async def send_context_update(self, 
                                 update_type: str, 
                                 data: Dict[str, Any],
                                 scope: ContextScope = ContextScope.GLOBAL,
                                 priority: ContextPriority = ContextPriority.NORMAL,
                                 target_modules: Optional[List[str]] = None):
        """
        Send a context update to other modules
        
        Args:
            update_type: Type of the update
            data: Update data
            scope: Scope of the update
            priority: Priority level
            target_modules: Specific target modules (for targeted scope)
        """
        if hasattr(self, '_context_system'):
            update = ContextUpdate(
                source_module=self.module_name,
                update_type=update_type,
                data=data,
                scope=scope,
                priority=priority,
                target_modules=target_modules
            )
            await self._context_system.add_context_update(update)
    
    async def get_cross_module_messages(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get messages from other modules"""
        if hasattr(self, '_context_system'):
            return await self._context_system.get_cross_module_communication(self.module_name)
        return {}
    
    def set_context_system(self, context_system: ContextDistributionSystem):
        """Set the context distribution system reference"""
        self._context_system = context_system
    
    # Processing stage methods - override these for coordinated processing
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input stage - override in subclasses"""
        return {}
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Process analysis stage - override in subclasses"""
        return {}
    
    async def process_integration(self, context: SharedContext) -> Dict[str, Any]:
        """Process integration stage - override in subclasses"""
        return {}
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Process synthesis stage - override in subclasses"""
        return {}
    
    async def finalize_context_session(self, context: SharedContext):
        """Finalize the context session - override to cleanup"""
        self.current_context = None

class EnhancedNyxBrainMixin:
    """
    Mixin to enhance NyxBrain with context distribution capabilities
    Add this to your NyxBrain class via multiple inheritance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_distribution = None
        self._module_registry: Dict[str, ContextAwareModule] = {}
    
    async def initialize_context_system(self):
        """Initialize the context distribution system"""
        if not hasattr(self, 'context_distribution') or not self.context_distribution:
            self.context_distribution = ContextDistributionSystem(self)
            
            # Register context-aware modules
            await self._register_context_aware_modules()
            
            logger.info("Context distribution system initialized")
            
    async def _register_context_aware_modules(self):
        """Register modules that support context distribution"""
        # Store list of modules that should be registered when created
        self._pending_modules = [
            'emotional_core', 'memory_core', 'reflection_engine', 'reasoning_core',
            'knowledge_core', 'goal_manager', 'agentic_action_generator', 
            'identity_evolution', 'needs_system', 'mood_manager', 'mode_integration',
            'passive_observation_system', 'proactive_communication_engine',
            'imagination_simulator', 'attentional_controller', 'relationship_manager',
            'theory_of_mind', 'digital_somatosensory_system', 'femdom_coordinator',
            'multimodal_integrator', 'spatial_mapper', 'agent_enhanced_memory',
            'hormone_system', 'reward_system', 'temporal_perception',
            'procedural_memory_manager', 'reflexive_system', 'meta_core'
        ]
        
        # Try to register any modules that already exist (unlikely at early init)
        for module_name in self._pending_modules[:]:  # Use slice to avoid modifying during iteration
            if hasattr(self, module_name) and getattr(self, module_name) is not None:
                module = getattr(self, module_name)
                await self.register_module_with_context_system(module_name, module)
                self._pending_modules.remove(module_name)
        
        logger.info(f"Context distribution system initialized. {len(self._pending_modules)} modules pending registration")
    
    async def register_module_with_context_system(self, module_name: str, module: Any) -> Any:
        """Register a module with the context system after it's created"""
        if not self.context_distribution:
            logger.warning(f"Cannot register {module_name} - context distribution not initialized")
            return module
            
        if not module:
            logger.warning(f"Cannot register {module_name} - module is None")
            return module
        
        # Remove from pending list if present
        if hasattr(self, '_pending_modules') and module_name in self._pending_modules:
            self._pending_modules.remove(module_name)
        
        # Check if already registered
        if module_name in self._module_registry:
            logger.debug(f"Module {module_name} already registered")
            return self._module_registry[module_name]
        
        # Wrap or register the module
        if not isinstance(module, ContextAwareModule):
            wrapped_module = self._wrap_module_for_context(module, module_name)
            self._module_registry[module_name] = wrapped_module
            logger.debug(f"Wrapped and registered module: {module_name}")
            return wrapped_module
        else:
            self._module_registry[module_name] = module
            if hasattr(module, 'set_context_system'):
                module.set_context_system(self.context_distribution)
            logger.debug(f"Registered context-aware module: {module_name}")
            return module

    def get_available_modules(self) -> Dict[str, Any]:
        """
        Get available modules for discovery by context-aware components
        
        Returns:
            Dictionary of available modules
        """
        available = {}
        
        # Return modules from the internal registry if available
        if hasattr(self, 'internal_module_registry'):
            available.update(self.internal_module_registry)
        
        # Also include modules from the context-aware registry
        if hasattr(self, '_module_registry'):
            for name, module in self._module_registry.items():
                if name not in available:
                    available[name] = {
                        "module": module,
                        "capabilities": getattr(module, 'capabilities', []),
                        "purposes": getattr(module, 'purposes', [])
                    }
        
        return available
    
    def _wrap_module_for_context(self, module, module_name: str) -> ContextAwareModule:
        """Wrap an existing module to make it context-aware"""
        
        class WrappedModule(ContextAwareModule):
            def __init__(self, original_module, name):
                super().__init__(name)
                self.original_module = original_module
                
                # Copy attributes from original module
                for attr_name in dir(original_module):
                    if not attr_name.startswith('_') and not hasattr(self, attr_name):
                        setattr(self, attr_name, getattr(original_module, attr_name))
            
            async def process_input(self, context: SharedContext) -> Dict[str, Any]:
                """Default input processing"""
                result = {}
                
                # Try common input processing methods
                if hasattr(self.original_module, 'process_input'):
                    try:
                        result = await self.original_module.process_input(
                            context.user_input, context.session_context
                        )
                    except Exception as e:
                        logger.error(f"Error in {self.module_name} process_input: {e}")
                        result = {"error": str(e)}
                
                # Send context updates if module generated them
                if isinstance(result, dict):
                    await self._send_relevant_updates(result)
                
                return result
            
            async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
                """Default analysis processing"""
                result = {}
                
                # Try analysis methods
                analysis_methods = [
                    'analyze', 'process_analysis', 'run_analysis', 
                    'evaluate', 'assess', 'examine'
                ]
                
                for method_name in analysis_methods:
                    if hasattr(self.original_module, method_name):
                        try:
                            method = getattr(self.original_module, method_name)
                            if callable(method):
                                result = await method(context.user_input, context.session_context)
                                break
                        except Exception as e:
                            logger.error(f"Error in {self.module_name} {method_name}: {e}")
                
                await self._send_relevant_updates(result)
                return result
            
            async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
                """Default synthesis processing"""
                result = {}
                
                # Get cross-module messages for synthesis
                messages = await self.get_cross_module_messages()
                
                # Try synthesis methods
                synthesis_methods = [
                    'synthesize', 'integrate', 'combine', 'generate_response'
                ]
                
                for method_name in synthesis_methods:
                    if hasattr(self.original_module, method_name):
                        try:
                            method = getattr(self.original_module, method_name)
                            if callable(method):
                                # Pass context and cross-module messages
                                result = await method(context, messages)
                                break
                        except Exception as e:
                            logger.error(f"Error in {self.module_name} {method_name}: {e}")
                
                return result
            
            async def _send_relevant_updates(self, result: Dict[str, Any]):
                """Send relevant context updates based on module results"""
                if not isinstance(result, dict):
                    return
                
                # Map module types to context update patterns
                update_mappings = {
                    'emotional_core': {
                        'context_key': 'emotional_state',
                        'data_keys': ['emotional_state', 'emotions', 'mood', 'valence', 'arousal']
                    },
                    'memory_core': {
                        'context_key': 'memory_context', 
                        'data_keys': ['memories', 'memory_id', 'retrieved_memories', 'memory_count']
                    },
                    'goal_manager': {
                        'context_key': 'goal_context',
                        'data_keys': ['goals', 'active_goals', 'goal_status', 'next_step']
                    },
                    'relationship_manager': {
                        'context_key': 'relationship_context',
                        'data_keys': ['relationship_state', 'trust', 'intimacy', 'user_profile']
                    },
                    'attentional_controller': {
                        'context_key': 'attention_context',
                        'data_keys': ['focus', 'attention_weight', 'inhibited_targets', 'current_foci']
                    },
                    'mode_integration': {
                        'context_key': 'mode_context',
                        'data_keys': ['mode', 'interaction_mode', 'mode_adjustments']
                    }
                }
                
                mapping = update_mappings.get(self.module_name)
                if mapping:
                    update_data = {}
                    for key in mapping['data_keys']:
                        if key in result:
                            update_data[key] = result[key]
                    
                    if update_data:
                        await self.send_context_update(
                            update_type=f"{self.module_name}_update",
                            data={mapping['context_key']: update_data}
                        )
        
        return WrappedModule(module, module_name)
    
    async def process_input_coordinated(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced process_input using context distribution system
        
        Args:
            user_input: User's input text
            context: Additional context
            
        Returns:
            Coordinated processing results
        """
        if not self.context_distribution:
            await self.initialize_context_system()
        
        # Initialize context session
        shared_context = await self.context_distribution.initialize_context_session(
            user_input=user_input,
            user_id=getattr(self, 'user_id', None),
            initial_context=context or {}
        )
        
        try:
            # Stage 1: Input Processing
            logger.debug("Stage 1: Input Processing")
            input_results = await self.context_distribution.coordinate_processing_stage("input")
            
            # Stage 2: Analysis Processing  
            logger.debug("Stage 2: Analysis Processing")
            analysis_results = await self.context_distribution.coordinate_processing_stage("analysis")
            
            # Stage 3: Integration Processing
            logger.debug("Stage 3: Integration Processing")
            integration_results = await self.context_distribution.coordinate_processing_stage("integration")
            
            # Return coordinated results
            return {
                "input_processing": input_results,
                "analysis_processing": analysis_results, 
                "integration_processing": integration_results,
                "shared_context": shared_context.dict(),
                "active_modules": list(shared_context.active_modules),
                "context_updates": len(shared_context.context_updates)
            }
            
        except Exception as e:
            logger.error(f"Error in coordinated input processing: {e}")
            return {"error": str(e), "fallback": await self._fallback_process_input(user_input, context)}
        
    async def generate_response_coordinated(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced generate_response using context distribution system
        
        Args:
            user_input: User's input text
            context: Additional context
            
        Returns:
            Coordinated response generation results
        """
        if not self.context_distribution:
            await self.initialize_context_system()
        
        # Process input first if not already done
        if not self.context_distribution.current_context:
            await self.process_input_coordinated(user_input, context)
        
        try:
            # Stage 4: Synthesis Processing
            logger.debug("Stage 4: Synthesis Processing")
            synthesis_results = await self.context_distribution.coordinate_processing_stage("synthesis")
            
            # Stage 5: Final Response Synthesis
            logger.debug("Stage 5: Final Response Synthesis")
            final_response = await self.context_distribution.synthesize_responses()
            
            # Finalize context session
            await self.context_distribution.finalize_context_session()
            
            return {
                "message": final_response.get("primary_response", "I'm processing your input."),
                "synthesis_results": synthesis_results,
                "final_synthesis": final_response,
                "coordination_metadata": final_response.get("synthesis_metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error in coordinated response generation: {e}")
            return {
                "message": "I encountered an issue while coordinating my response.",
                "error": str(e),
                "fallback": await self._fallback_generate_response(user_input, context)
            }
    
    async def _fallback_process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback to original process_input method"""
        if hasattr(super(), 'process_input'):
            return await super().process_input(user_input, context)
        return {"fallback_used": True, "user_input": user_input}
    
    async def _fallback_generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback to original generate_response method"""
        if hasattr(super(), 'generate_response'):
            return await super().generate_response(user_input, context)
        return {"message": "I need to process that further.", "fallback_used": True}
    
    def get_context_distribution_status(self) -> Dict[str, Any]:
        """Get status of the context distribution system"""
        if not self.context_distribution:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "registered_modules": len(self._module_registry),
            "active_context": self.context_distribution.current_context is not None,
            "context_history_length": len(self.context_distribution.context_history),
            "registered_module_names": list(self._module_registry.keys())
        }
