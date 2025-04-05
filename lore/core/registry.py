# lore/core/registry.py

from typing import Dict, Any, List, Type, Optional, Callable
import logging
import importlib
import inspect

from agents import (
    Agent, function_tool, Runner, trace, GuardrailFunctionOutput, 
    InputGuardrail, OutputGuardrail, AgentHooks, handoff
)
from agents.run import RunConfig

class ManagerRegistry:
    """
    Enhanced registry for all manager classes with lazy loading, dependency injection,
    agent tools, and handoff system.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._managers = {}
        self._class_map = {}  # This will be populated in __init__.py
        self._function_tools = {}  # Registry of function tools
        self._trace_group = f"registry_{user_id}_{conversation_id}"
        self._metadata = {
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "component": "ManagerRegistry"
        }
        
        # Initialize orchestrator agent
        self._init_orchestrator()
    
    def _init_orchestrator(self):
        """Initialize the orchestrator agent for cross-manager coordination."""
        self.orchestrator = Agent(
            name="ManagerOrchestrator",
            instructions=(
                "You coordinate operations across multiple specialized managers. "
                "You receive requests and determine which manager(s) should handle them, "
                "then orchestrate handoffs between managers when needed for complex tasks."
            ),
            model="o3-mini",
        )
        
        # We'll add handoffs as managers are registered
    
    @function_tool
    async def get_manager(self, manager_key: str) -> 'BaseLoreManager':
        """
        Get a manager instance by key, creating it if not already created.
        Will ensure the manager is initialized.
        
        Args:
            manager_key: Key of the manager to get
            
        Returns:
            Instance of the manager
        """
        with trace(
            "GetManager", 
            group_id=self._trace_group,
            metadata={**self._metadata, "manager_key": manager_key}
        ):
            if manager_key not in self._managers:
                if manager_key not in self._class_map:
                    raise ValueError(f"Unknown manager key: {manager_key}")
                    
                manager_class = self._class_map[manager_key]
                self._managers[manager_key] = manager_class(self.user_id, self.conversation_id)
                
                # Register new manager's function tools
                self._register_manager_tools(manager_key, self._managers[manager_key])
                
                # Add handoff to orchestrator if needed
                self._add_orchestrator_handoff(manager_key, self._managers[manager_key])
            
            # Ensure manager is initialized
            manager = self._managers[manager_key]
            await manager.ensure_initialized()
            return manager
    
    def _register_manager_tools(self, manager_key: str, manager_instance: 'BaseLoreManager'):
        """Register function tools from a manager instance."""
        for name, method in inspect.getmembers(manager_instance, predicate=inspect.ismethod):
            if hasattr(method, "_is_function_tool") and method._is_function_tool:
                tool_key = f"{manager_key}.{name}"
                self._function_tools[tool_key] = method
                logging.info(f"Registered tool: {tool_key}")
    
    def _add_orchestrator_handoff(self, manager_key: str, manager_instance: 'BaseLoreManager'):
        """Add a handoff to the orchestrator agent for the manager."""
        # This would normally define a handoff, but let's simplify for this example
        logging.info(f"Added handoff capability for {manager_key}")
    
    @function_tool
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available function tools across all managers.
        
        Returns:
            List of tool information dictionaries
        """
        result = []
        for tool_key, tool_func in self._function_tools.items():
            # Extract tool info from the function
            tool_info = {
                "key": tool_key,
                "name": tool_func.__name__,
                "description": tool_func.__doc__ or "No description available"
            }
            
            # Add parameter info if available
            params = []
            for param_name, param in inspect.signature(tool_func).parameters.items():
                if param_name not in ['self', 'ctx']:
                    param_info = {
                        "name": param_name,
                        "required": param.default == inspect.Parameter.empty
                    }
                    
                    # Add type annotation if available
                    if param.annotation != inspect.Parameter.empty:
                        param_info["type"] = str(param.annotation)
                    
                    params.append(param_info)
            
            tool_info["parameters"] = params
            result.append(tool_info)
        
        return result
    
    @function_tool
    async def discover_manager_relationships(self) -> Dict[str, List[str]]:
        """
        Map relationships between managers by analyzing their interaction patterns.
        
        Returns:
            Dictionary mapping manager keys to lists of related manager keys
        """
        relationships = {}
        
        # This would normally analyze actual interaction patterns
        # For this example, we'll use some predefined relationships
        relationships = {
            "geopolitical": ["world_politics", "regional_culture"],
            "world_politics": ["geopolitical", "religion"],
            "religion": ["world_politics", "local_lore"],
            "local_lore": ["religion", "regional_culture"],
            "regional_culture": ["local_lore", "geopolitical"],
            "educational": ["regional_culture", "religion"],
            "master": ["geopolitical", "world_politics", "religion", "local_lore", "regional_culture", "educational"]
        }
        
        return relationships
    
    @function_tool
    async def execute_cross_manager_handoff(
        self, 
        starting_manager: str, 
        target_manager: str, 
        operation: str, 
        params: Dict[str, Any]
    ) -> Any:
        """
        Execute an operation that requires handoff between managers.
        
        Args:
            starting_manager: Key of the manager initiating the handoff
            target_manager: Key of the manager receiving the handoff
            operation: Name of the operation to perform
            params: Parameters for the operation
            
        Returns:
            Result of the handoff operation
        """
        with trace(
            "CrossManagerHandoff", 
            group_id=self._trace_group,
            metadata={
                **self._metadata, 
                "starting_manager": starting_manager,
                "target_manager": target_manager,
                "operation": operation
            }
        ):
            # Get both managers
            start_manager = await self.get_manager(starting_manager)
            target_manager = await self.get_manager(target_manager)
            
            # Create run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "handoff_source": starting_manager,
                "handoff_target": target_manager,
                "operation": operation
            })
            
            # Create handoff prompt for orchestrator
            handoff_prompt = (
                f"Coordinate a handoff from {starting_manager} to {target_manager} "
                f"for operation '{operation}' with parameters: {json.dumps(params, indent=2)}"
            )
            
            # Execute the handoff through the orchestrator
            result = await Runner.run(
                self.orchestrator,
                handoff_prompt,
                context=run_ctx.context
            )
            
            # Extract the operation result from the orchestrator output
            # This assumes the orchestrator formats its response in a way we can parse
            try:
                return json.loads(result.final_output)
            except:
                return {"result": result.final_output}
    
    
    async def get_lore_dynamics(self):
        """Get the LoreDynamicsSystem instance"""
        return await self.get_manager('lore_dynamics')
    
    async def get_geopolitical_manager(self):
        """Get the GeopoliticalSystemManager instance"""
        return await self.get_manager('geopolitical')
    
    async def get_local_lore_manager(self):
        """Get the LocalLoreManager instance"""
        return await self.get_manager('local_lore')
    
    async def get_religion_manager(self):
        """Get the ReligionManager instance"""
        return await self.get_manager('religion')
    
    async def get_world_politics_manager(self):
        """Get the WorldPoliticsManager instance"""
        return await self.get_manager('world_politics')
    
    async def get_regional_culture_system(self):
        """Get the RegionalCultureSystem instance"""
        return await self.get_manager('regional_culture')
    
    async def get_educational_system_manager(self):
        """Get the EducationalSystemManager instance"""
        return await self.get_manager('educational')
    
    async def get_master_lore_system(self):
        """Get the MatriarchalLoreSystem instance"""
        return await self.get_manager('master')
    
    async def initialize_all(self):
        """Initialize all manager instances."""
        for key in self._class_map.keys():
            await self.get_manager(key)
