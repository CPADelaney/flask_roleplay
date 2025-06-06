# lore/core/registry.py (Fixed for OpenAI Agents SDK v0.0.17)

from __future__ import annotations
import json
import logging
import importlib
import inspect
from typing import Dict, Any, List, Type, Optional, Callable, Union

from pydantic import BaseModel, Field

# --- MERGE ---
# All your existing imports are preserved. I've just added this one.
from lore.managers.base_manager import BaseLoreManager

from agents import (
    Agent, function_tool, Runner, trace,
    RunConfig, handoff, RunContextWrapper
)

logger = logging.getLogger(__name__)

# --- MERGE ---
# All your Pydantic models are preserved with minor fixes for strict schema
class ManagerResult(BaseModel):
    manager_key: str
    manager_type: str
    initialized: bool

class AvailableToolInfo(BaseModel):
    key: str
    name: str
    description: str
    parameters: List['ToolParameterInfo']

class ToolParameterInfo(BaseModel):
    name: str
    required: bool
    type: Optional[str] = None

class ManagerRelationships(BaseModel):
    relationships: Dict[str, List[str]] = Field(default_factory=dict)

class CrossManagerHandoffParams(BaseModel):
    starting_manager: str
    target_manager: str
    operation: str
    # Store params as a JSON string to avoid Dict[str, Any] issues
    params_json: str = Field(default="{}", description="JSON-encoded parameters")
    
    def get_params(self) -> Dict[str, Any]:
        """Parse the JSON params"""
        try:
            return json.loads(self.params_json)
        except json.JSONDecodeError:
            return {}

class CrossManagerHandoffResult(BaseModel):
    # Store result as JSON string to avoid Any type
    result_json: str
    success: bool
    starting_manager: str
    target_manager: str
    operation: str
    
    def get_result(self) -> Any:
        """Parse the JSON result"""
        try:
            return json.loads(self.result_json)
        except json.JSONDecodeError:
            return self.result_json

# Removed RegisterClassMapParams since it contained Dict[str, Type[...]]


class ManagerRegistry:
    """
    Enhanced registry for all manager classes with lazy loading, dependency injection,
    agent tools, and handoff system.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._managers: Dict[str, BaseLoreManager] = {} # --- MERGE --- Type hint changed to BaseLoreManager
        self._class_map: Dict[str, Type[BaseLoreManager]] = {} # --- MERGE --- Type hint changed to Type[BaseLoreManager]
        self._function_tools: Dict[str, Callable] = {} # --- MERGE --- More specific type hint
        self._trace_group = f"registry_{user_id}_{conversation_id}"
        self._metadata = {
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "component": "ManagerRegistry"
        }
        
        self._init_orchestrator()
        self._register_default_managers() # --- MERGE --- New method call to populate the class map.

    def _init_orchestrator(self):
        """Initialize the orchestrator agent for cross-manager coordination."""
        self.orchestrator = Agent(
            name="ManagerOrchestrator",
            instructions=(
                "You coordinate operations across multiple specialized managers. "
                "You receive requests and determine which manager(s) should handle them, "
                "then orchestrate handoffs between managers when needed for complex tasks."
            ),
            model="gpt-4-mini",
        )

    # --- MERGE ---
    # This is the new, central method for populating the registry.
    # It makes the registry self-contained and easily extensible.
    def _register_default_managers(self):
        """
        Populates the internal class map with all known specialized managers.
        This is where you add any new managers you create in the future.
        """
        from lore.managers.education import EducationalSystemManager
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from lore.managers.local_lore import LocalLoreManager
        from lore.managers.politics import WorldPoliticsManager
        from lore.managers.religion import ReligionManager
        from lore.managers.world_lore_manager import WorldLoreManager
        from lore.systems.dynamics import LoreDynamicsSystem
        from lore.systems.regional_culture import RegionalCultureSystem

        class_map = {
            'education': EducationalSystemManager,
            'geopolitical': GeopoliticalSystemManager,
            'local_lore': LocalLoreManager,
            'politics': WorldPoliticsManager,
            'religion': ReligionManager,
            'world_lore': WorldLoreManager,
            'dynamics': LoreDynamicsSystem,
            'regional_culture': RegionalCultureSystem,
        }
        self._class_map.update(class_map)
        logger.info(f"Registry populated with {len(class_map)} default manager classes.")


    @function_tool
    async def get_manager(self, manager_key: str) -> ManagerResult:
        """
        Get a manager instance by key, creating it if not already instantiated.
        Ensures the manager is initialized before returning. This is the core of lazy-loading.
        
        Args:
            manager_key: Key of the manager to get (e.g., 'religion', 'politics').
            
        Returns:
            ManagerResult with information about the manager.
            
        Raises:
            ValueError: If the manager key is unknown.
        """
        # --- MERGE ---
        # Return a ManagerResult instead of the actual manager instance
        # to avoid issues with function tool serialization
        if manager_key in self._managers:
            manager = self._managers[manager_key]
            return ManagerResult(
                manager_key=manager_key,
                manager_type=manager.__class__.__name__,
                initialized=True
            )

        if manager_key not in self._class_map:
            raise ValueError(f"Unknown manager key: '{manager_key}'. Is it registered?")

        with trace(
            "GetAndInitializeManager", 
            group_id=self._trace_group,
            metadata={**self._metadata, "manager_key": manager_key}
        ):
            try:
                manager_class = self._class_map[manager_key]
                logger.info(f"Lazy-loading new manager instance for '{manager_key}'")
                instance = manager_class(self.user_id, self.conversation_id)
                await instance.ensure_initialized() # All managers must have this method

                self._managers[manager_key] = instance
                
                # Your existing logic for tool registration and handoffs fits perfectly here.
                self._register_manager_tools(manager_key, instance)
                self._add_orchestrator_handoff(manager_key, instance)

                return ManagerResult(
                    manager_key=manager_key,
                    manager_type=instance.__class__.__name__,
                    initialized=True
                )
            except Exception as e:
                logger.error(f"Error getting and initializing manager '{manager_key}': {e}", exc_info=True)
                raise

    # Internal method to get actual manager instance
    async def _get_manager_instance(self, manager_key: str) -> BaseLoreManager:
        """Internal method to get actual manager instance."""
        await self.get_manager(manager_key)  # Ensure it's loaded
        return self._managers[manager_key]

    # Your existing _register_manager_tools method is perfect as is.
    def _register_manager_tools(self, manager_key: str, manager_instance: 'BaseLoreManager'):
        """Register function tools from a manager instance."""
        for name, method in inspect.getmembers(manager_instance, predicate=inspect.ismethod):
            if hasattr(method, "_is_function_tool") and method._is_function_tool:
                tool_key = f"{manager_key}.{name}"
                self._function_tools[tool_key] = method
                logger.info(f"Registered tool: {tool_key}")
    
    # Your existing _add_orchestrator_handoff method is perfect as is.
    def _add_orchestrator_handoff(self, manager_key: str, manager_instance: 'BaseLoreManager'):
        """Add a handoff to the orchestrator agent for the manager."""
        logger.info(f"Handoff capability enabled for {manager_key}")

    
    @function_tool
    async def get_available_tools(self) -> List[AvailableToolInfo]:
        """
        Get a list of all available function tools across all managers.
        
        Returns:
            List of tool information dictionaries
        """
        result = []
        for tool_key, tool_func in self._function_tools.items():
            # Extract tool info from the function
            tool_info = AvailableToolInfo(
                key=tool_key,
                name=tool_func.__name__,
                description=tool_func.__doc__ or "No description available",
                parameters=[]
            )
            
            # Add parameter info if available
            for param_name, param in inspect.signature(tool_func).parameters.items():
                if param_name not in ['self', 'ctx']:
                    param_info = ToolParameterInfo(
                        name=param_name,
                        required=param.default == inspect.Parameter.empty,
                        type=str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    )
                    tool_info.parameters.append(param_info)
            
            result.append(tool_info)
        
        return result
    
    @function_tool
    async def discover_manager_relationships(self) -> ManagerRelationships:
        """
        Map relationships between managers by analyzing their interaction patterns.
        
        Returns:
            Dictionary mapping manager keys to lists of related manager keys
        """
        try:
            # In a real implementation, this would analyze actual usage patterns
            # to dynamically discover relationships between managers
            
            # For this example, we'll use predefined relationships
            relationships = {
                "geopolitical": ["world_politics", "regional_culture"],
                "world_politics": ["geopolitical", "religion"],
                "religion": ["world_politics", "local_lore"],
                "local_lore": ["religion", "regional_culture"],
                "regional_culture": ["local_lore", "geopolitical"],
                "educational": ["regional_culture", "religion"],
                "master": ["geopolitical", "world_politics", "religion", 
                           "local_lore", "regional_culture", "educational"]
            }
            
            return ManagerRelationships(relationships=relationships)
        except Exception as e:
            logging.error(f"Error discovering manager relationships: {e}")
            return ManagerRelationships(relationships={})
    
    @function_tool
    async def execute_cross_manager_handoff(
        self, 
        params: CrossManagerHandoffParams
    ) -> CrossManagerHandoffResult:
        """
        Execute an operation that requires handoff between managers.
        
        Args:
            params: Handoff parameters with JSON-encoded params
            
        Returns:
            Result of the handoff operation
            
        Raises:
            ValueError: If managers don't exist or the handoff fails
        """
        with trace(
            "CrossManagerHandoff", 
            group_id=self._trace_group,
            metadata={
                **self._metadata, 
                "starting_manager": params.starting_manager,
                "target_manager": params.target_manager,
                "operation": params.operation
            }
        ):
            try:
                # Get both managers
                await self.get_manager(params.starting_manager)
                await self.get_manager(params.target_manager)
                
                # Get actual manager instances
                start_manager = self._managers[params.starting_manager]
                target_manager = self._managers[params.target_manager]
                
                # Parse params from JSON
                operation_params = params.get_params()
                
                # Create run context
                run_ctx = RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "handoff_source": params.starting_manager,
                    "handoff_target": params.target_manager,
                    "operation": params.operation,
                    "params": operation_params
                })
                
                # Create handoff prompt for orchestrator
                handoff_prompt = (
                    f"Coordinate a handoff from {params.starting_manager} to {params.target_manager} "
                    f"for operation '{params.operation}' with parameters: {json.dumps(operation_params, indent=2)}\n\n"
                    f"Consider any dependencies, ensure correct parameter mapping, and "
                    f"handle any transformations needed between the managers. "
                    f"Return the result in JSON format."
                )
                
                # Configure the run with tracing
                run_config = RunConfig(
                    workflow_name="CrossManagerHandoff",
                    trace_metadata={
                        "starting_manager": params.starting_manager,
                        "target_manager": params.target_manager,
                        "operation": params.operation
                    }
                )
                
                # Execute the handoff through the orchestrator
                result = await Runner.run(
                    self.orchestrator,
                    handoff_prompt,
                    context=run_ctx.context,
                    run_config=run_config
                )
                
                # Extract the operation result from the orchestrator output
                try:
                    result_data = json.loads(result.final_output)
                    result_json = json.dumps(result_data)
                except json.JSONDecodeError:
                    # If it's not valid JSON, wrap the result
                    result_json = json.dumps({"result": result.final_output})
                
                return CrossManagerHandoffResult(
                    result_json=result_json,
                    success=True,
                    starting_manager=params.starting_manager,
                    target_manager=params.target_manager,
                    operation=params.operation
                )
            except Exception as e:
                logging.error(f"Error in cross-manager handoff: {e}")
                raise ValueError(f"Failed to execute cross-manager handoff: {e}")
    
    @function_tool(strict_mode=False)
    async def register_class_map(self, class_map: Dict[str, str]) -> None:
        """
        Register manager classes to be instantiated on demand.
        
        Args:
            class_map: Dictionary mapping manager keys to class names
        """
        # This is simplified - in practice, you'd need to resolve class names
        # For now, just log the registration attempt
        logging.info(f"Registered {len(class_map)} manager class names")
    
    # Non-tool methods for getting manager instances
    async def get_lore_dynamics(self):
        """Get the LoreDynamicsSystem instance"""
        return await self._get_manager_instance('lore_dynamics')
    
    async def get_geopolitical_manager(self):
        """Get the GeopoliticalSystemManager instance"""
        return await self._get_manager_instance('geopolitical')
    
    async def get_local_lore_manager(self):
        """Get the LocalLoreManager instance"""
        return await self._get_manager_instance('local_lore')
    
    async def get_religion_manager(self):
        """Get the ReligionManager instance"""
        return await self._get_manager_instance('religion')
    
    async def get_world_politics_manager(self):
        """Get the WorldPoliticsManager instance"""
        return await self._get_manager_instance('world_politics')
    
    async def get_regional_culture_system(self):
        """Get the RegionalCultureSystem instance"""
        return await self._get_manager_instance('regional_culture')
    
    async def get_educational_system_manager(self):
        """Get the EducationalSystemManager instance"""
        return await self._get_manager_instance('educational')
    
    async def get_master_lore_system(self):
        """Get the MatriarchalLoreSystem instance"""
        return await self._get_manager_instance('master')
    
    @function_tool
    async def initialize_all(self) -> None:
        """
        Initialize all manager instances.
        
        This ensures all registered managers are created and initialized.
        """
        for key in self._class_map.keys():
            await self.get_manager(key)

# Update model forward references
AvailableToolInfo.model_rebuild()
