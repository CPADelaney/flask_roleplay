# lore/core/registry.py – Refactored with dynamic imports
from __future__ import annotations

import importlib
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Type, Optional, Tuple

from agents import Agent, Runner, RunConfig, function_tool, trace
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────
class ManagerResult(BaseModel):
    """Result of getting a manager instance."""
    manager_key: str
    manager_type: str
    initialized: bool


class ToolParameterInfo(BaseModel):
    """Information about a tool parameter."""
    name: str
    required: bool
    type: str | None = None


class AvailableToolInfo(BaseModel):
    """Information about an available tool."""
    key: str
    name: str
    description: str
    parameters: List[ToolParameterInfo]


class ManagerRelationships(BaseModel):
    """Mapping of manager relationships."""
    relationships: Dict[str, List[str]] = Field(default_factory=dict)


class CrossManagerHandoffParams(BaseModel):
    """Parameters for cross-manager handoff operations."""
    starting_manager: str
    target_manager: str
    operation: str
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation parameters to pass between managers"
    )


class CrossManagerHandoffResult(BaseModel):
    """Result of a cross-manager handoff operation."""
    result: Any
    success: bool
    starting_manager: str
    target_manager: str
    operation: str
    error: str | None = None


# ──────────────────────────────────────────────────────────────
# Manager Registry
# ──────────────────────────────────────────────────────────────
class ManagerRegistry:
    """
    Central registry for all Lore manager classes.
    
    Features:
    - Dynamic imports to avoid circular dependencies
    - Lazy loading of manager instances  
    - Automatic tool registration
    - Cross-manager handoff coordination
    - Ordered initialization support
    - Dependency tracking and relationships
    """
    
    # Manager configuration: key -> (module_path, class_name)
    MANAGER_CONFIG: Dict[str, Tuple[str, str]] = {
        'education': ('lore.managers.education', 'EducationalSystemManager'),
        'geopolitical': ('lore.managers.geopolitical', 'GeopoliticalSystemManager'),
        'local_lore': ('lore.managers.local_lore', 'LocalLoreManager'),
        'politics': ('lore.managers.politics', 'WorldPoliticsManager'),
        'religion': ('lore.managers.religion', 'ReligionManager'),
        'world_lore': ('lore.managers.world_lore_manager', 'WorldLoreManager'),
        'lore_dynamics': ('lore.systems.dynamics', 'LoreDynamicsSystem'),
        'regional_culture': ('lore.systems.regional_culture', 'RegionalCultureSystem'),
    }
    
    # Initialization order for dependencies
    INIT_ORDER = [
        'world_lore',        # Foundation
        'regional_culture',  # Cultural bedrock
        'lore_dynamics',     # System dynamics
        'education',
        'religion', 
        'politics',
        'geopolitical',
        'local_lore',        # Depends on locations
    ]
    
    # Default manager relationships
    DEFAULT_RELATIONSHIPS = {
        "geopolitical": ["politics", "regional_culture", "lore_dynamics"],
        "politics": ["geopolitical", "religion", "lore_dynamics"],
        "religion": ["politics", "local_lore", "regional_culture"],
        "local_lore": ["religion", "regional_culture", "world_lore"],
        "regional_culture": ["local_lore", "geopolitical", "education"],
        "education": ["regional_culture", "religion", "local_lore"],
        "world_lore": ["geopolitical", "politics", "religion", "local_lore"],
        "lore_dynamics": ["geopolitical", "politics", "world_lore"],
    }

    def __init__(self, user_id: int, conversation_id: int) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id

        # Core state
        self._managers: Dict[str, Any] = {}
        self._function_tools: Dict[str, Callable] = {}
        self._relationships: Dict[str, List[str]] = self.DEFAULT_RELATIONSHIPS.copy()
        
        # Manager configuration (can be extended at runtime)
        self._manager_config = self.MANAGER_CONFIG.copy()
        self._init_order = self.INIT_ORDER.copy()

        # Tracing metadata
        self._trace_group = f"registry_{user_id}_{conversation_id}"
        self._metadata = {
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "component": "ManagerRegistry",
        }

        # Initialize orchestrator for cross-manager coordination
        self._init_orchestrator()

    # ─── Core Internal Methods ────────────────────────────────
    def _init_orchestrator(self) -> None:
        """Initialize the orchestrator agent for cross-manager coordination."""
        self.orchestrator = Agent(
            name="ManagerOrchestrator",
            instructions=(
                "You coordinate operations across multiple specialized managers. "
                "Analyze requests to determine which manager(s) should handle them, "
                "and orchestrate handoffs between managers for complex tasks. "
                "Always consider manager relationships and dependencies."
            ),
            model="gpt-5-nano",
        )

    async def _get_or_create_manager(self, manager_key: str) -> Any:
        """
        Get existing or create new manager instance using dynamic imports.
        
        Args:
            manager_key: Key identifying the manager
            
        Returns:
            The initialized manager instance
        """
        if manager_key in self._managers:
            return self._managers[manager_key]

        if manager_key not in self._manager_config:
            raise ValueError(f"Unknown manager key: {manager_key}")

        with trace(
            "GetAndInitializeManager",
            group_id=self._trace_group,
            metadata={**self._metadata, "manager_key": manager_key},
        ):
            try:
                # Dynamic import to avoid circular dependencies
                module_name, class_name = self._manager_config[manager_key]
                module = importlib.import_module(module_name)
                manager_class = getattr(module, class_name)
                
                logger.info(f"Lazy-loading manager: {manager_key} from {module_name}.{class_name}")
                
                # Create and initialize manager
                manager = manager_class(self.user_id, self.conversation_id)
                
                # Call ensure_initialized if available (backward compatibility)
                if hasattr(manager, 'ensure_initialized'):
                    await manager.ensure_initialized()
                
                # Store and configure
                self._managers[manager_key] = manager
                self._register_manager_tools(manager_key, manager)
                self._add_orchestrator_handoff(manager_key, manager)
                
                logger.info(f"Successfully initialized manager: {manager_key}")
                return manager
                
            except Exception as e:
                logger.error(f"Failed to initialize manager {manager_key}: {e}", exc_info=True)
                raise

    def _register_manager_tools(self, manager_key: str, manager: Any) -> None:
        """Register all function tools from a manager instance."""
        tool_count = 0
        for name, attr in inspect.getmembers(manager):
            if isinstance(attr, FunctionTool):
                full_key = f"{manager_key}.{attr.name or name}"
                self._function_tools[full_key] = attr
                tool_count += 1
                logger.debug(f"Registered tool: {full_key}")
        
        if tool_count > 0:
            logger.info(f"Registered {tool_count} tools for manager: {manager_key}")

    def _add_orchestrator_handoff(self, manager_key: str, manager: Any) -> None:
        """Add handoff capability to the orchestrator for a manager."""
        manager_tools = [tool for key, tool in self._function_tools.items() 
                        if key.startswith(f"{manager_key}.")]
        if manager_tools:
            self.orchestrator.tools.extend(manager_tools)
            logger.debug(f"Added {len(manager_tools)} tools from {manager_key} to orchestrator")

    # ─── Public API Methods (exposed as tools) ────────────────
    @function_tool
    async def get_manager(self, manager_key: str) -> ManagerResult:
        """
        Get metadata about a manager, initializing it if needed.
        
        Args:
            manager_key: The key identifying the manager
            
        Returns:
            ManagerResult with information about the initialized manager
        """
        manager = await self._get_or_create_manager(manager_key)
        return ManagerResult(
            manager_key=manager_key,
            manager_type=manager.__class__.__name__,
            initialized=True,
        )

    @function_tool
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered managers in dependency order.
        
        Returns:
            Dictionary mapping manager keys to initialization success status
        """
        results = {}
        
        # Use initialization order for known managers
        for key in self._init_order:
            if key in self._manager_config:
                try:
                    await self._get_or_create_manager(key)
                    results[key] = True
                except Exception as e:
                    logger.error(f"Failed to initialize {key}: {e}")
                    results[key] = False
        
        # Initialize any additional managers not in init order
        for key in self._manager_config:
            if key not in results:
                try:
                    await self._get_or_create_manager(key)
                    results[key] = True
                except Exception as e:
                    logger.error(f"Failed to initialize {key}: {e}")
                    results[key] = False
                    
        return results

    @function_tool
    async def get_available_tools(self) -> List[AvailableToolInfo]:
        """
        Get information about all available tools across all managers.
        
        Returns:
            List of available tools with their metadata
        """
        tools: List[AvailableToolInfo] = []
        
        for key, tool in self._function_tools.items():
            fn = tool.function if hasattr(tool, 'function') else tool
            sig = inspect.signature(fn)
            
            params = [
                ToolParameterInfo(
                    name=p.name,
                    required=p.default is inspect.Parameter.empty,
                    type=str(p.annotation) if p.annotation != inspect.Parameter.empty else None,
                )
                for p in sig.parameters.values()
                if p.name not in {"self", "ctx"}
            ]
            
            description = tool.description if hasattr(tool, 'description') else (fn.__doc__ or "No description")
            
            tools.append(AvailableToolInfo(
                key=key,
                name=tool.name if hasattr(tool, 'name') else fn.__name__,
                description=description.strip().split('\n')[0],
                parameters=params,
            ))
        
        return sorted(tools, key=lambda t: t.key)

    @function_tool
    async def discover_manager_relationships(self) -> ManagerRelationships:
        """
        Get the relationship map between managers.
        
        Returns:
            ManagerRelationships showing which managers are related
        """
        return ManagerRelationships(relationships=self._relationships.copy())

    @function_tool(strict_mode=False)
    async def execute_cross_manager_handoff(
        self, params: CrossManagerHandoffParams
    ) -> CrossManagerHandoffResult:
        """
        Execute an operation that requires coordination between managers.
        
        Args:
            params: Parameters specifying the handoff operation
            
        Returns:
            Result of the cross-manager operation
        """
        with trace(
            "CrossManagerHandoff",
            group_id=self._trace_group,
            metadata={
                **self._metadata,
                "handoff": {
                    "from": params.starting_manager,
                    "to": params.target_manager,
                    "operation": params.operation,
                }
            },
        ):
            try:
                # Ensure both managers are initialized
                start_mgr = await self._get_or_create_manager(params.starting_manager)
                target_mgr = await self._get_or_create_manager(params.target_manager)
                
                # Create context for the handoff
                handoff_context = {
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "source_manager": params.starting_manager,
                    "target_manager": params.target_manager,
                    "operation": params.operation,
                    "params": params.params,
                }
                
                # Execute through orchestrator
                run_config = RunConfig(
                    workflow_name="CrossManagerHandoff",
                    trace_metadata=handoff_context,
                )
                
                prompt = (
                    f"Coordinate handoff from {params.starting_manager} to {params.target_manager} "
                    f"for operation: {params.operation}\n"
                    f"Parameters: {json.dumps(params.params, indent=2)}\n\n"
                    f"Ensure proper data transformation and parameter mapping between managers."
                )
                
                result = await Runner.run(
                    self.orchestrator,
                    prompt,
                    context=handoff_context,
                    run_config=run_config,
                )
                
                return CrossManagerHandoffResult(
                    result=result.final_output,
                    success=True,
                    starting_manager=params.starting_manager,
                    target_manager=params.target_manager,
                    operation=params.operation,
                )
                
            except Exception as e:
                logger.error(f"Handoff failed: {e}", exc_info=True)
                return CrossManagerHandoffResult(
                    result=None,
                    success=False,
                    starting_manager=params.starting_manager,
                    target_manager=params.target_manager,
                    operation=params.operation,
                    error=str(e),
                )

    # ─── Configuration Methods ────────────────────────────────
    def register_manager(self, key: str, module_path: str, class_name: str) -> None:
        """
        Register a new manager configuration.
        
        Args:
            key: Unique key for the manager
            module_path: Python module path containing the manager class
            class_name: Name of the manager class
        """
        if key in self._manager_config:
            logger.warning(f"Overwriting existing manager registration: {key}")
        
        self._manager_config[key] = (module_path, class_name)
        logger.info(f"Registered manager: {key} -> {module_path}.{class_name}")

    def register_relationship(self, manager_key: str, related_managers: List[str]) -> None:
        """
        Register relationships for a manager.
        
        Args:
            manager_key: The manager to set relationships for
            related_managers: List of related manager keys
        """
        self._relationships[manager_key] = related_managers
        logger.info(f"Registered relationships for {manager_key}: {related_managers}")

    def set_initialization_order(self, order: List[str]) -> None:
        """
        Set custom initialization order for managers.
        
        Args:
            order: List of manager keys in initialization order
        """
        self._init_order = order
        logger.info(f"Updated initialization order: {order}")

    # ─── Convenience Methods (backward compatibility) ─────────
    async def get_lore_dynamics(self) -> Any:
        """Get the LoreDynamicsSystem instance."""
        return await self._get_or_create_manager("lore_dynamics")

    async def get_geopolitical_manager(self) -> Any:
        """Get the GeopoliticalSystemManager instance."""
        return await self._get_or_create_manager("geopolitical")

    async def get_local_lore_manager(self) -> Any:
        """Get the LocalLoreManager instance."""
        return await self._get_or_create_manager("local_lore")

    async def get_religion_manager(self) -> Any:
        """Get the ReligionManager instance."""
        return await self._get_or_create_manager("religion")

    async def get_politics_manager(self) -> Any:
        """Get the WorldPoliticsManager instance."""
        return await self._get_or_create_manager("politics")

    async def get_regional_culture_system(self) -> Any:
        """Get the RegionalCultureSystem instance."""
        return await self._get_or_create_manager("regional_culture")

    async def get_education_manager(self) -> Any:
        """Get the EducationalSystemManager instance."""
        return await self._get_or_create_manager("education")

    async def get_world_lore_manager(self) -> Any:
        """Get the WorldLoreManager instance."""
        return await self._get_or_create_manager("world_lore")


# Rebuild models to ensure forward references are resolved
AvailableToolInfo.model_rebuild()
