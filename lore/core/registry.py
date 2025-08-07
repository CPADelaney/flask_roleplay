# lore/core/registry.py – compatible with Agents SDK ≥0.1.0
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Type, Optional

from agents import Agent, Runner, RunConfig, function_tool, trace
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from lore.managers.base_manager import BaseLoreManager

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
    - Lazy loading of manager instances
    - Automatic tool registration
    - Cross-manager handoff coordination
    - Dependency tracking and relationships
    """

    def __init__(self, user_id: int, conversation_id: int) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id

        self._managers: Dict[str, BaseLoreManager] = {}
        self._class_map: Dict[str, Type[BaseLoreManager]] = {}
        self._function_tools: Dict[str, Callable] = {}
        self._relationships: Dict[str, List[str]] = {}

        self._trace_group = f"registry_{user_id}_{conversation_id}"
        self._metadata = {
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "component": "ManagerRegistry",
        }

        self._init_orchestrator()
        self._register_default_managers()
        self._register_default_relationships()

    # ─── Core Internal Methods ────────────────────────────────
    async def _get_or_init_manager(self, manager_key: str) -> BaseLoreManager:
        """
        Get or initialize a manager instance.
        
        This is the core method that implements lazy loading.
        All other methods should use this to access managers.
        """
        if manager_key in self._managers:
            return self._managers[manager_key]

        if manager_key not in self._class_map:
            raise ValueError(f"Unknown manager key: {manager_key}")

        with trace(
            "GetAndInitializeManager",
            group_id=self._trace_group,
            metadata={**self._metadata, "manager_key": manager_key},
        ):
            try:
                mgr_cls = self._class_map[manager_key]
                logger.info(f"Lazy-loading manager: {manager_key}")
                
                mgr = mgr_cls(self.user_id, self.conversation_id)
                await mgr.ensure_initialized()

                self._managers[manager_key] = mgr
                self._register_manager_tools(manager_key, mgr)
                self._add_orchestrator_handoff(manager_key, mgr)
                
                logger.info(f"Successfully initialized manager: {manager_key}")
                return mgr
                
            except Exception as e:
                logger.error(f"Failed to initialize manager {manager_key}: {e}", exc_info=True)
                raise

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

    def _register_default_managers(self) -> None:
        """Register all default manager classes."""
        # Import locally to avoid circular dependencies
        from lore.managers.education import EducationalSystemManager
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from lore.managers.local_lore import LocalLoreManager
        from lore.managers.politics import WorldPoliticsManager
        from lore.managers.religion import ReligionManager
        from lore.managers.world_lore_manager import WorldLoreManager
        from lore.systems.dynamics import LoreDynamicsSystem
        from lore.systems.regional_culture import RegionalCultureSystem

        self._class_map.update({
            "education": EducationalSystemManager,
            "geopolitical": GeopoliticalSystemManager,
            "local_lore": LocalLoreManager,
            "politics": WorldPoliticsManager,
            "religion": ReligionManager,
            "world_lore": WorldLoreManager,
            "lore_dynamics": LoreDynamicsSystem,
            "regional_culture": RegionalCultureSystem,
        })
        
        logger.info(f"Registered {len(self._class_map)} default manager classes")

    def _register_default_relationships(self) -> None:
        """Register default relationships between managers."""
        self._relationships = {
            "geopolitical": ["politics", "regional_culture", "lore_dynamics"],
            "politics": ["geopolitical", "religion", "lore_dynamics"],
            "religion": ["politics", "local_lore", "regional_culture"],
            "local_lore": ["religion", "regional_culture", "world_lore"],
            "regional_culture": ["local_lore", "geopolitical", "education"],
            "education": ["regional_culture", "religion", "local_lore"],
            "world_lore": ["geopolitical", "politics", "religion", "local_lore"],
            "lore_dynamics": ["geopolitical", "politics", "world_lore"],
        }

    def _register_manager_tools(self, manager_key: str, mgr: BaseLoreManager) -> None:
        """Register all function tools from a manager instance."""
        tool_count = 0
        for name, attr in inspect.getmembers(mgr):
            if isinstance(attr, FunctionTool):
                full_key = f"{manager_key}.{attr.name or name}"
                self._function_tools[full_key] = attr
                tool_count += 1
                logger.debug(f"Registered tool: {full_key}")
        
        if tool_count > 0:
            logger.info(f"Registered {tool_count} tools for manager: {manager_key}")

    def _add_orchestrator_handoff(self, manager_key: str, mgr: BaseLoreManager) -> None:
        """Add handoff capability to the orchestrator for a manager."""
        # Add manager's tools to orchestrator's available tools
        manager_tools = [tool for key, tool in self._function_tools.items() 
                        if key.startswith(f"{manager_key}.")]
        if manager_tools:
            self.orchestrator.tools.extend(manager_tools)
            logger.debug(f"Added {len(manager_tools)} tools from {manager_key} to orchestrator")
        
        logger.debug(f"Handoff capability enabled for: {manager_key}")

    # ─── Public API Methods (exposed as tools) ────────────────
    @function_tool
    async def get_manager(self, manager_key: str) -> ManagerResult:
        """
        Get metadata about a manager, initializing it if needed.
        
        Args:
            manager_key: The key identifying the manager (e.g., 'religion', 'politics')
            
        Returns:
            ManagerResult with information about the initialized manager
        """
        mgr = await self._get_or_init_manager(manager_key)
        return ManagerResult(
            manager_key=manager_key,
            manager_type=mgr.__class__.__name__,
            initialized=True,
        )

    @function_tool
    async def get_available_tools(self) -> List[AvailableToolInfo]:
        """
        Get information about all available tools across all managers.
        
        Returns:
            List of available tools with their metadata
        """
        tools: List[AvailableToolInfo] = []
        
        for key, tool in self._function_tools.items():
            # FunctionTool objects have their own function attribute
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
            
            # Get description from tool or function docstring
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
                start_mgr = await self._get_or_init_manager(params.starting_manager)
                target_mgr = await self._get_or_init_manager(params.target_manager)
                
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
                
                # Note: In SDK 0.1.0, the parameter is 'context'. 
                # In future versions, it may change to 'run_context'.
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

    @function_tool
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered managers.
        
        Returns:
            Dictionary mapping manager keys to initialization success status
        """
        results = {}
        for key in self._class_map:
            try:
                await self._get_or_init_manager(key)
                results[key] = True
            except Exception as e:
                logger.error(f"Failed to initialize {key}: {e}")
                results[key] = False
        return results

    # ─── Convenience Methods (not exposed as tools) ───────────
    async def get_lore_dynamics(self) -> BaseLoreManager:
        """Get the LoreDynamicsSystem instance."""
        return await self._get_or_init_manager("lore_dynamics")

    async def get_geopolitical_manager(self) -> BaseLoreManager:
        """Get the GeopoliticalSystemManager instance."""
        return await self._get_or_init_manager("geopolitical")

    async def get_local_lore_manager(self) -> BaseLoreManager:
        """Get the LocalLoreManager instance."""
        return await self._get_or_init_manager("local_lore")

    async def get_religion_manager(self) -> BaseLoreManager:
        """Get the ReligionManager instance."""
        return await self._get_or_init_manager("religion")

    async def get_politics_manager(self) -> BaseLoreManager:
        """Get the WorldPoliticsManager instance."""
        return await self._get_or_init_manager("politics")

    async def get_regional_culture_system(self) -> BaseLoreManager:
        """Get the RegionalCultureSystem instance."""
        return await self._get_or_init_manager("regional_culture")

    async def get_education_manager(self) -> BaseLoreManager:
        """Get the EducationalSystemManager instance."""
        return await self._get_or_init_manager("education")

    async def get_world_lore_manager(self) -> BaseLoreManager:
        """Get the WorldLoreManager instance."""
        return await self._get_or_init_manager("world_lore")

    # ─── Manager Registration Methods ─────────────────────────
    def register_manager(self, key: str, manager_class: Type[BaseLoreManager]) -> None:
        """
        Register a new manager class.
        
        Args:
            key: Unique key for the manager
            manager_class: The manager class to register
        """
        if key in self._class_map:
            logger.warning(f"Overwriting existing manager registration: {key}")
        
        self._class_map[key] = manager_class
        logger.info(f"Registered manager: {key} -> {manager_class.__name__}")

    def register_relationship(self, manager_key: str, related_managers: List[str]) -> None:
        """
        Register relationships for a manager.
        
        Args:
            manager_key: The manager to set relationships for
            related_managers: List of related manager keys
        """
        self._relationships[manager_key] = related_managers
        logger.info(f"Registered relationships for {manager_key}: {related_managers}")


# Rebuild models to ensure forward references are resolved
AvailableToolInfo.model_rebuild()
